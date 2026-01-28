import asyncio
from prebuild.amem_new import construct_memory
from pathlib import Path
import json
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from config.config import MEMORY_SYSTEMS_DIR
import time
from prebuild.usage_tracker import reset_usage, get_and_reset
from collections import defaultdict
from datetime import datetime

# -------------------------
# 收集 q_id
# -------------------------
def collect_targets(base_dir="/data/new"):
    targets = []
    seen = set()
    for p in Path(base_dir).glob("processed*.jsonl"):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = (
                        obj.get("_id")
                    )
                    if qid and qid not in seen:
                        seen.add(qid)
                        targets.append(qid)
        except Exception as e:
            print(f"Skipping file {p}: {e}")
    return targets


TARGETS = collect_targets()


def _init_logger():
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        log_dir / "build_samples.log",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    from datetime import datetime
    import os
    import json
    from config.config import (
        LLM_BASE_URL, MODELS, LLM_GENERATION, RESPONSE_FORMAT,
        MEMORY_SYSTEMS_DIR, PDF_TMP_DIR, DATASETS, MAX_CONCURRENCY, SAVE_CHECKS, MINERU
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mem_tag = os.path.basename(MEMORY_SYSTEMS_DIR.rstrip("/")) or "memory_systems"

    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "LLM_BASE_URL": LLM_BASE_URL,
        "MODELS": MODELS,
        "LLM_GENERATION": LLM_GENERATION,
        "RESPONSE_FORMAT_keys": list(RESPONSE_FORMAT.keys()),
        "MEMORY_SYSTEMS_DIR": MEMORY_SYSTEMS_DIR,
        "PDF_TMP_DIR": PDF_TMP_DIR,
        "DATASETS_keys": list(DATASETS.keys()),
        "MAX_CONCURRENCY": MAX_CONCURRENCY,
        "SAVE_CHECKS": {k: getattr(SAVE_CHECKS, k) for k in vars(SAVE_CHECKS)},
        "MINERU": MINERU,
        "api_key_masked": "****"
    }

    snapshot_path = log_dir / f"{mem_tag}_config_{timestamp}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    logger.info(f"Config snapshot saved: {snapshot_path}")
    return logger


def analyze_tokens_per_minute(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze usage data and calculate tokens per minute"""
    if not usage or "calls" not in usage or not usage["calls"]:
        return {
            "max_tokens_per_minute": 0,
            "tokens_by_minute": {},
            "total_duration_minutes": 0
        }
    
    calls = usage["calls"]
    tokens_by_minute = defaultdict(int)
    
    # 找到最早和最晚的时间戳
    min_timestamp = None
    max_timestamp = None
    
    for call in calls:
        if "timestamp" not in call:
            continue
        
        ts = call["timestamp"]
        if min_timestamp is None or ts < min_timestamp:
            min_timestamp = ts
        if max_timestamp is None or ts > max_timestamp:
            max_timestamp = ts
        
        # 计算这个调用属于哪一分钟（从开始时间算起）
        if min_timestamp is not None:
            minute_offset = int((ts - min_timestamp) / 60)
            
            # 获取这次调用的token数
            tokens = call.get("total_tokens", 0)
            tokens_by_minute[minute_offset] += tokens
    
    if not tokens_by_minute:
        return {
            "max_tokens_per_minute": 0,
            "tokens_by_minute": {},
            "total_duration_minutes": 0
        }
    
    max_tokens = max(tokens_by_minute.values())
    max_minute = max(tokens_by_minute.keys(), key=lambda k: tokens_by_minute[k])
    
    duration_minutes = (max_timestamp - min_timestamp) / 60 if max_timestamp and min_timestamp else 0
    
    # 转换为可序列化的格式
    tokens_by_minute_serializable = {f"minute_{k}": v for k, v in sorted(tokens_by_minute.items())}
    
    return {
        "max_tokens_per_minute": max_tokens,
        "max_tokens_minute_offset": max_minute,
        "tokens_by_minute": tokens_by_minute_serializable,
        "total_duration_minutes": round(duration_minutes, 2),
        "total_calls": len(calls)
    }


# -------------------------
# 在子进程中执行 1 个任务
# -------------------------
def _build_one_run(qid: str) -> Tuple[str, bool, Any]:
    try:
        logger.info(f"[子进程] 开始构建: {qid}")
        reset_usage()
        start_t = time.time()
        ms = asyncio.run(construct_memory(qid, evolve_iters=3, window_size=3))
        duration_sec = round(time.time() - start_t, 3)
        usage = get_and_reset()

        # 分析每分钟的token使用量
        tokens_per_minute_stats = analyze_tokens_per_minute(usage)

        # 写入每个 sample 自己的路径下
        sample_dir = Path(MEMORY_SYSTEMS_DIR) / qid
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取分阶段统计信息
        by_stage = usage.get("by_stage", {})
        
        report = {
            "qid": qid,
            "duration_sec": duration_sec,
            "memory_count": len(ms.memories),
            # 总体统计
            "overall_usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "embedding_tokens": usage.get("embedding_tokens", 0),
            },
            # 分阶段统计
            "stage_wise_usage": {
                "text_analysis": by_stage.get("text_analysis", {}),
                "image_analysis": by_stage.get("image_analysis", {}),
                "text_embedding": by_stage.get("text_embedding", {}),
                "image_embedding": by_stage.get("image_embedding", {}),
                "memory_evolution": by_stage.get("memory_evolution", {}),
                "re_embedding": by_stage.get("re_embedding", {}),
            },
            # 完整的usage数据（包含calls详情）
            "full_usage": usage,
            "tokens_per_minute_stats": tokens_per_minute_stats,
            "artifacts": {
                "memories_pkl": str(sample_dir / "memories.pkl"),
                "retriever_embeddings_npy": str(sample_dir / "retriever_embeddings.npy"),
            },
        }
        build_json_path = sample_dir / "build.json"
        if build_json_path.exists():
            logger.info(f"[子进程] 检测到 {build_json_path} 已存在，跳过写入。")
        else:
            try:
                # 独占创建，避免竞态覆盖（如果并发时两次同时创建）
                with build_json_path.open("x", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except FileExistsError:
                logger.info(f"[子进程] 检测到 {build_json_path} 已存在，跳过写入。")

        # 生成分阶段日志信息
        stage_logs = []
        for stage_name, stage_data in by_stage.items():
            if stage_data.get("calls", 0) > 0:
                tokens_info = []
                if "total_tokens" in stage_data:
                    tokens_info.append(f"tokens={stage_data['total_tokens']}")
                if "embedding_tokens" in stage_data:
                    tokens_info.append(f"embed_tokens={stage_data['embedding_tokens']}")
                tokens_info.append(f"calls={stage_data['calls']}")
                tokens_info.append(f"time={stage_data.get('duration_sec', 0):.2f}s")
                stage_logs.append(f"{stage_name}({', '.join(tokens_info)})")
        
        logger.info(
            f"[子进程] 构建完成: {qid}, 记忆数: {len(ms.memories)}, "
            f"总tokens: {usage.get('total_tokens', 0)}, "
            f"总embedding_tokens: {usage.get('embedding_tokens', 0)}, "
            f"总时长: {duration_sec}s"
        )
        if stage_logs:
            logger.info(f"[子进程] 分阶段统计: {' | '.join(stage_logs)}")
        return qid, True, {
            "memory_count": len(ms.memories), 
            "metrics": {
                "qid": qid, 
                "duration_sec": duration_sec,
                "overall_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "embedding_tokens": usage.get("embedding_tokens", 0),
                },
                "stage_wise_usage": by_stage,
                "tokens_per_minute_stats": tokens_per_minute_stats
            }
        }
    except Exception as e:
        logger.error(f"[子进程] 构建失败: {qid}: {e}")
        return qid, False, str(e)


# -------------------------
# chunk 切分（保留）
# -------------------------
def _chunk_targets(targets: List[str], size: int) -> List[List[str]]:
    buf = []
    out = []
    for t in targets:
        buf.append(t)
        if len(buf) >= size:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


# -------------------------
# 主函数：多进程串行 + 无线程
# -------------------------
def main():
    _init_logger()

    PROC_WORKERS = 4
    CHUNK_SIZE = 1  # 一个 chunk 就一个任务（强烈推荐保留）

    logger.info(f"总任务数: {len(TARGETS)}, 进程数: {PROC_WORKERS}, chunk 大小: {CHUNK_SIZE}")

    chunks = _chunk_targets(TARGETS, CHUNK_SIZE)

    ok = 0
    fail = 0
    all_metrics = []

    with tqdm(total=len(TARGETS), desc="Prebuilding", ncols=100) as pbar:
        with ProcessPoolExecutor(max_workers=PROC_WORKERS) as pool:
            fut_map = {
                pool.submit(_build_one_run, chunk[0]): chunk
                for chunk in chunks
            }

            for fut in as_completed(fut_map):
                chunk = fut_map[fut]
                qid = chunk[0]

                try:
                    t, success, extra = fut.result()
                    if success:
                        ok += 1
                        if isinstance(extra, dict) and "metrics" in extra:
                            all_metrics.append(extra["metrics"])
                    else:
                        fail += 1
                        logger.error(f"失败: {qid}: {extra}")
                except Exception as e:
                    fail += 1
                    logger.error(f"进程执行失败: {qid}: {e}")
                pbar.update(1)
    
    logger.info(f"完成：成功 {ok}，失败 {fail}，总计 {len(TARGETS)}")
    print(f"\n完成：成功 {ok}，失败 {fail}，总计 {len(TARGETS)}")


if __name__ == "__main__":
    main()