import os
import json
import time
import pickle
import asyncio
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI,AsyncOpenAI

from config.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    EMBED_BASE_URL,
    EMBED_API_KEY,
    MODELS,
    LLM_GENERATION,
    RESPONSE_FORMAT,
    PROMPTS,
    MEMORY_SYSTEMS_DIR,
    PDF_TMP_DIR,
    DATASETS,
    MAX_CONCURRENCY,
    SAVE_CHECKS,
    MINERU,
    PARALLEL_ANALYSIS,
)

from prebuild.memory_layer import AgenticMemorySystem
from prebuild.visdom_utils import (
    extract_text_from_pdf,
    split_text,
    extract_images_from_pdf,
    encode_image,
    clean_text,
    get_pdf,
)
from prebuild.mineru_utils import extract_chunk_from_mineru, extract_image_from_mineru

from prebuild.usage_tracker import add_chat_usage, add_embed_usage, add_stage_duration, add_single_call_duration
from tenacity import retry, stop_after_attempt, wait_exponential
# Optional: tqdm_asyncio is nice to have; if not available, fallback to asyncio.gather
try:
    from tqdm.asyncio import tqdm_asyncio
    _use_tqdm = True
except Exception:
    _use_tqdm = False


# -----------------------------
# Client
# -----------------------------
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
aclient = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

qwen_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
qwen_aclient = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


_llm_sem_by_loop = {}

def _get_llm_semaphore():
    loop = asyncio.get_running_loop()
    sem = _llm_sem_by_loop.get(id(loop))
    if sem is None:
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        _llm_sem_by_loop[id(loop)] = sem
    return sem

# -----------------------------
# Utilities
# -----------------------------
async def _gather(tasks: List[asyncio.Task]):
    if _use_tqdm:
        return await tqdm_asyncio.gather(*tasks)
    return await asyncio.gather(*tasks)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_actual_file(base_dir: str, filename: str) -> str | None:
    """Robust matching for potentially mangled filenames (encoding issues)."""
    direct_path = os.path.join(base_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    try:
        if not os.path.exists(base_dir):
            print(f"Warning: directory does not exist: {base_dir}")
            return None
        all_files = os.listdir(base_dir)
        filename_base, filename_ext = os.path.splitext(filename)
        for actual in all_files:
            ab, ae = os.path.splitext(actual)
            if ae.lower() != filename_ext.lower():
                continue
            if ab.lower() == filename_base.lower():
                return os.path.join(base_dir, actual)
            ne = filename_base.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            na = ab.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            if na.lower() == ne.lower():
                return os.path.join(base_dir, actual)
        from difflib import SequenceMatcher
        best, score = None, 0.8
        for actual in all_files:
            if os.path.splitext(actual)[1].lower() == filename_ext.lower():
                r = SequenceMatcher(None, filename.lower(), actual.lower()).ratio()
                if r > score:
                    best, score = actual, r
        if best:
            print(f"找到相似文件: {filename} -> {best} (相似度: {score:.2%})")
            return os.path.join(base_dir, best)
        print(f"警告：未找到文件: {filename}")
        return None
    except Exception as e:
        print(f"查找文件时出错 {filename}: {e}")
        return None


# -----------------------------
# Dataset helpers
# -----------------------------
_dataset_cache: Dict[str, pd.DataFrame] = {}

def load_dataset_df(name: str) -> pd.DataFrame:
    cfg = DATASETS[name]
    if name in _dataset_cache:
        return _dataset_cache[name]
    enc = cfg.get("encoding", "utf-8")
    df = pd.read_csv(cfg["csv"], encoding=enc)
    _dataset_cache[name] = df
    return df


def resolve_docs_from_dataset(dataset_name: str, q_id: str, limit: int = 5) -> Tuple[str, List[str]]:
    cfg = DATASETS[dataset_name]
    df = load_dataset_df(dataset_name)
    key, docs_col = cfg["key"], cfg["docs_col"]
    matches = np.where(df[key] == q_id)[0]
    if len(matches) == 0:
        raise ValueError(
            f"错误：在 {dataset_name}.csv 中找不到 {key}='{q_id}' 的数据。\n"
            f"可用的前若干 {key} 值：{df[key].unique()[:10].tolist()}..."
        )
    row = df.iloc[matches[0]].to_dict()
    base_dir = cfg["base_dir"]
    doc_names = list(eval(row[docs_col]))[:limit]
    pdf_paths: List[str] = []
    for doc in doc_names:
        p = find_actual_file(base_dir, doc)
        if p:
            pdf_paths.append(p)
        else:
            print(f"跳过无法找到的文件: {doc}")
    print(f"成功找到 {len(pdf_paths)}/{len(doc_names)} 个文件")
    return base_dir, pdf_paths

# Mineru dir aggregation
def resolve_docs_from_dataset_mineru(dataset_name: str, q_id: str, limit: int = 5) -> Tuple[str, List[str]]:
    cfg = DATASETS[dataset_name]
    df = load_dataset_df(dataset_name)
    key, docs_col = cfg["key"], cfg["docs_col"]

    matches = np.where(df[key] == q_id)[0]
    if len(matches) == 0:
        raise ValueError(
            f"错误：在 {dataset_name}.csv 中找不到 {key}='{q_id}' 的数据。\n"
            f"可用的前若干 {key} 值：{df[key].unique()[:10].tolist()}..."
        )

    row = df.iloc[matches[0]].to_dict()
    base_dir = cfg["mineru_dir"]

    # ============================
    # ✨ 去掉 .pdf 后缀（你需要的部分）
    # ============================
    doc_names_raw = list(eval(row[docs_col]))[:limit]
    doc_names = [os.path.splitext(d)[0] for d in doc_names_raw]
    # ============================

    mineru_paths: List[str] = []

    for doc in doc_names:
        p = find_actual_file(base_dir, doc)

        # 深入两层
        for _ in range(2):
            subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subs) == 1:
                p = os.path.join(p, subs[0])
            else:
                break

        if p:
            mineru_paths.append(p)
        else:
            print(f"跳过无法找到的文件: {doc}")

    print(f"成功找到 {len(mineru_paths)}/{len(doc_names)} 个文件")
    return base_dir, mineru_paths

def _log_failed_response(raw: str, error: Exception, is_multimodal: bool, user_payload):
    """将解析失败的 LLM 响应记录到日志文件"""
    from datetime import datetime
    log_dir = os.path.join(MEMORY_SYSTEMS_DIR, "_debug_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(log_dir, f"failed_response_{timestamp}.txt")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"时间: {datetime.now().isoformat()}\n")
        f.write(f"类型: {'multimodal' if is_multimodal else 'text'}\n")
        f.write(f"错误: {type(error).__name__}: {error}\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write("【原始响应 (raw response)】\n")
        f.write("-" * 40 + "\n")
        f.write(raw if raw else "(空响应)")
        f.write("\n" + "-" * 40 + "\n\n")
        
        f.write(f"响应长度: {len(raw) if raw else 0} 字符\n")
        f.write(f"finish_reason: (见上方错误信息)\n\n")
        
        # 记录输入（对于文本类型）
        if not is_multimodal and isinstance(user_payload, str):
            f.write("【输入内容 (user_payload 前2000字符)】\n")
            f.write("-" * 40 + "\n")
            f.write(user_payload[:2000])
            f.write("\n" + "-" * 40 + "\n")
    
    print(f"[DEBUG] 解析失败的响应已保存到: {log_file}")
    return log_file


async def call_llm_json(system: str, user_payload, *, is_multimodal: bool = False):
    sem = _get_llm_semaphore()
    async with sem:
        call_start = time.time()  # 记录单次调用开始时间
        try:
            resp = await qwen_aclient.chat.completions.create(
                    model=MODELS["chat"],
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={"type": "json_object"},
                    # frequency_penalty=0.5,
                    **LLM_GENERATION,
                )
        except Exception as err:
                # network / timeout / Azure filtering
            print(f"[ERROR] LLM request failed: {err}")
            await asyncio.sleep(1)
            raise  # ❌ 抛出异常，不记录此次调用时间

        # ✅ 只有成功的调用才计算耗时
        call_duration = time.time() - call_start
        
        try:
            qkind = "analyze_multimodal" if is_multimodal else "analyze_text"
            add_chat_usage(
                getattr(resp, "usage", None),
                {"model": MODELS["chat"], "qkind": qkind}
            )
            # 只记录成功调用的最大耗时
            stage = "image_analysis" if is_multimodal else "text_analysis"
            add_single_call_duration(stage, call_duration)
        except Exception:
            pass 

        raw = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        
        # 检查是否被截断
        if finish_reason == "length":
            print(f"[WARNING] 响应被截断 (finish_reason=length)，可能导致 JSON 不完整")
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"{e}")
            # 尝试提取 JSON 对象
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError as e2:
                    # 记录失败的响应
                    _log_failed_response(raw, e2, is_multimodal, user_payload)
                    raise
            # 记录失败的响应
            _log_failed_response(raw, e, is_multimodal, user_payload)
            raise
    

async def analyze_content(payload: str, *, modality: str) -> Dict[str, any]:
    """Unified analyzer for text or image.
    - modality: 'text' (payload is plain text) or 'image' (payload is base64 string)
    """
    system = "You must respond with a JSON object."
    if modality == "text":
        content = clean_text(payload)
        user = PROMPTS["text_keyword"] + content
        return await call_llm_json(system, user)
    elif modality == "image":
        user = [
            {"type": "text", "text": PROMPTS["image_keyword"]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
        ]
        return await call_llm_json(system, user, is_multimodal=True)
    else:
        raise ValueError("modality must be 'text' or 'image'")

async def analyze_content_mineru(payload: str, *, modality: str, context: str = "", caption: str = "") -> Dict[str, any]:
    """Unified analyzer for text or image.
    - modality: 'text' (payload is plain text) or 'image' (payload is base64 string)
    """
    system = "You must respond with a JSON object."
    if modality == "text":
        content = clean_text(payload)
        user = PROMPTS["text"] + content
        return await call_llm_json(system, user)
    elif modality == "image":
        prompt  = PROMPTS["image_ocr_keyword"].replace("{context}", context).replace("{caption}", caption)
        user = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}},
        ]

        return await call_llm_json(system, user, is_multimodal=True)
    else:
        raise ValueError("modality must be 'text' or 'image'")
# -----------------------------
# Embedding helpers
# -----------------------------
async def embed_one(text: str, kind: str = "embedding") -> List[float]:
    # 使用单独的 Embedding 异步客户端
    call_start = time.time()  # 记录单次调用开始时间
    try:
        resp = await embed_aclient.embeddings.create(model=MODELS["embed"], input=text)
    except Exception as err:
        # ❌ API 失败，不记录此次调用时间
        print(f"[ERROR] Embedding request failed: {err}")
        raise
    
    # ✅ 只有成功的调用才计算耗时
    call_duration = time.time() - call_start
    
    try:
        add_embed_usage(getattr(resp, "usage", None), {"model": MODELS["embed"], "kind": kind})
        # 只记录成功调用的最大耗时
        add_single_call_duration(kind, call_duration)
    except Exception:
        pass
    return resp.data[0].embedding  # type: ignore


async def embed_many(texts: List[str], kind: str = "embedding") -> List[List[float]]:
    # simple concurrency control to avoid overwhelming server
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _task(t: str):
        async with sem:
            return await embed_one(t, kind=kind)

    return await asyncio.gather(*[_task(t) for t in texts])


# -----------------------------
# Public API (main entry points)
# -----------------------------
async def construct_memory(
    pdf_path: str,
    *,
    evolve_iters: int = 1,
    window_size: int = 2,
) -> AgenticMemorySystem:
    """
    Build the memory system for a given q_id (dataset) or a remote/local PDF path.

    Keeps visdom_utils & memory_layer imports intact.
    Uses config for URLs, API keys, absolute paths, and prompts.
    """
    print("\n" + "=" * 80)
    print(f"开始构建Memory System: {pdf_path}")
    print("=" * 80)

    ensure_dir(MEMORY_SYSTEMS_DIR)
    ensure_dir(PDF_TMP_DIR)

    # Detect dataset by substring; else treat as URL/local file and download
    detected_dataset = next((n for n in DATASETS.keys() if n in pdf_path), None)

    flag_dataset = detected_dataset is not None
    flag_mineru = MINERU 
    name = pdf_path  # use provided key/q_id as memory name for dataset cases
    if flag_dataset:
        if flag_mineru:
            _, pdf_paths = resolve_docs_from_dataset_mineru(detected_dataset, pdf_path)
        else:
            _, pdf_paths = resolve_docs_from_dataset(detected_dataset, pdf_path)
        print(pdf_paths)
    else:
        print("Downloading pdf")
        filename = time.strftime("%Y%m%d_%H%M%S") + ".pdf"
        out_path = os.path.join(PDF_TMP_DIR, filename)
        try:
            await get_pdf(pdf_path, out_path)
        except Exception as e:
            print(f"Error downloading pdf: {e}")
            raise
        pdf_paths = [out_path]

    # Load or initialize memory system
    existing = set(os.listdir(MEMORY_SYSTEMS_DIR))
    if name in existing:
        print(f"Loading existing memory system: {name}")
        ms = AgenticMemorySystem(model_name=MODELS["embed"], llm_model=MODELS["chat"])  # type: ignore
        ms.load_memory_system(name+"_iter_"+str(evolve_iters))
    else:
        print("initializing memory system")
        ms = AgenticMemorySystem(model_name=MODELS["embed"], llm_model=MODELS["chat"])  # type: ignore

        if not flag_mineru:
            # --- Extract text & images ---
            pages = []
            for p in pdf_paths:
                pages.extend(extract_text_from_pdf(p))#针对每个pdf提取
            chunks = []
            for page in pages:
                chunks.extend(split_text(page)) #针对每个page分chunk
             
            images = []
            for pdf_path in pdf_paths:
                images.extend(extract_images_from_pdf(pdf_path))  
             
            images_b64 = [encode_image(image) for image in images]
        else:
            chunks = []
            images_b64 = []
            contexts = []
            captions = []
            for p in pdf_paths:
                chunks.extend(extract_chunk_from_mineru(p))  
                imgs, context,caption = extract_image_from_mineru(p)
                images_b64.extend(encode_image(img) for img in imgs)
                contexts.extend(context)
                captions.extend(caption)


        # Analyze也要改 TODO
        
        # ============================================================
        # 定义文本处理函数（分析 + 嵌入）
        # ============================================================
        async def process_text_content():
            """处理所有文本内容：分析 + 嵌入"""
            print(f"Analyzing textual content ({len(chunks)} chunks)...")
            text_analysis_start = time.time()
            text_tasks = [analyze_content(ch, modality="text") for ch in chunks]
             
            try:
                text_results = await asyncio.gather(*text_tasks, return_exceptions=True)
            except Exception as e:
                print(f"错误：文本分析批量处理失败: {e}")
                raise
            text_analysis_duration = time.time() - text_analysis_start
            add_stage_duration("text_analysis", text_analysis_duration)
            
            text_contents: Dict[int, Dict[str, Any]] = {}
            failed = 0
            for i, r in enumerate(text_results):
                if isinstance(r, Exception) or not isinstance(r, dict) or "summary" not in r:
                    print(r)
                    failed += 1
                    text_contents[i] = {"summary": "内容分析失败", "keywords": ["未知"], "tags": ["错误"]}
                else:
                    text_contents[i] = r
            if failed:
                print(f"警告：共有 {failed}/{len(chunks)} 个文本块分析失败，使用默认值填充")
            if len(text_contents) == 0:
                raise RuntimeError("❌ 严重错误：所有文本块分析均失败！")
            print(f"文本分析完成: {len(text_contents) - failed}/{len(chunks)} 成功")
             
            # Embed text
            print("Embedding textual content")
            text_embedding_start = time.time()
            text_docs = [
                f"{text_contents[i]['summary']} keywords: {', '.join(text_contents[i]['keywords'])}"
                for i in range(len(text_contents))
            ]
            try:
                text_vecs = await embed_many(text_docs, kind="text_embedding")
                for i, emb in enumerate(text_vecs):
                    text_contents[i]["embedding"] = [emb]
                print(f"文本嵌入完成: {len(text_vecs)} 个向量")
            except Exception as e:
                print(f"错误：文本嵌入生成失败: {e}")
                raise
            text_embedding_duration = time.time() - text_embedding_start
            add_stage_duration("text_embedding", text_embedding_duration)
            
            return text_contents
        
        # ============================================================
        # 定义图像处理函数（分析 + 嵌入）
        # ============================================================
        async def process_image_content():
            """处理所有图像内容：分析 + 嵌入"""
            print(f"Analyzing visual content ({len(images_b64)} images)...")
            image_contents: Dict[int, Dict[str, Any]] = {}
            
            if not images_b64:
                print("跳过图像分析：未从PDF中提取到图像")
                return image_contents
            
            image_analysis_start = time.time()
            if not MINERU:
                img_tasks = [analyze_content(b64, modality="image") for b64 in images_b64]
            else:
                img_tasks = [analyze_content_mineru(b64, modality="image", context=contexts[i], caption=captions[i]) for i, b64 in enumerate(images_b64)]
            
            try:
                img_results = await asyncio.gather(*img_tasks, return_exceptions=True)
            except Exception as e:
                print(f"错误：图像分析批量处理失败: {e}")
                raise
            image_analysis_duration = time.time() - image_analysis_start
            add_stage_duration("image_analysis", image_analysis_duration)
            
            img_failed = 0
            for i, r in enumerate(img_results):
                if isinstance(r, Exception) or not isinstance(r, dict) or "summary" not in r:
                    img_failed += 1
                    image_contents[i] = {"summary": "图像分析失败", "keywords": ["未知"], "tags": ["错误"]}
                else:
                    image_contents[i] = r
            if img_failed:
                print(f"警告：共有 {img_failed}/{len(images_b64)} 个图像分析失败，使用默认值填充")
            if img_failed == len(images_b64):
                print("警告：所有图像分析均失败！将跳过图像notes添加。")
                return {}
            else:
                print(f"图像分析完成: {len(image_contents) - img_failed}/{len(images_b64)} 成功")
            
            # Embed images (if any valid)
            if image_contents:
                print("Embedding visual content")
                image_embedding_start = time.time()
                img_docs = [
                    f"{image_contents[i]['summary']} keywords: {', '.join(image_contents[i]['keywords'])}"
                    for i in range(len(image_contents))
                ]
                try:
                    img_vecs = await embed_many(img_docs, kind="image_embedding")
                    for i, emb in enumerate(img_vecs):
                        image_contents[i]["embedding"] = [emb]
                    print(f"图像嵌入完成: {len(img_vecs)} 个向量")
                except Exception as e:
                    print(f"错误：图像嵌入生成失败: {e}")
                    raise
                image_embedding_duration = time.time() - image_embedding_start
                add_stage_duration("image_embedding", image_embedding_duration)
            
            return image_contents
        
        if PARALLEL_ANALYSIS:
            print("🚀 使用并行分析模式（文本和图像同时处理）")
            parallel_start = time.time()
            text_contents, image_contents = await asyncio.gather(
                process_text_content(),
                process_image_content()
            )
            parallel_duration = time.time() - parallel_start
            print(f"✓ 并行处理完成，总耗时: {parallel_duration:.2f}秒")
        else:
            print("📋 使用串行分析模式（文本和图像依次处理）")
            text_contents = await process_text_content()
            image_contents = await process_image_content()
         
        # --- Add notes ---
        # filter out the notes with "No meaningful information" in the summary
        to_remove = [i for i, text_content in text_contents.items() if "No meaningful information" in text_content["summary"]]
        print(f"过滤掉 {len(to_remove)} 条包含 'No meaningful information' 的文本notes")
        
        print("Adding textual notes to memory system")
        ok, bad = 0, 0
        for i, ch in enumerate(chunks):
            if i in to_remove:
                continue
            try:
                ms.add_note(
                    content=ch,
                    context=text_contents[i]["summary"],
                    keywords=text_contents[i]["keywords"],
                    tags=text_contents[i]["tags"],
                    category="text",
                    pre_embeddings=text_contents[i]["embedding"],
                )
                ok += 1
            except Exception as e:
                bad += 1
                if bad <= 3:
                    import traceback; traceback.print_exc()
        print(f"文本notes添加完成: {ok} 成功, {bad} 失败")
        if ok == 0:
            raise RuntimeError("❌ 严重错误：未能成功添加任何文本note！")
         
        print("Adding visual notes to memory system")
        ok_i, bad_i = 0, 0
        for i, b64 in enumerate(images_b64):
            if i not in image_contents:
                continue
            try:
                ms.add_note(
                    content=b64,
                    context=image_contents[i]["summary"],
                    keywords=image_contents[i]["keywords"],
                    tags=image_contents[i]["tags"],
                    text_content = image_contents[i]["text_content"],
                    category="image",
                    visual=True,
                    pre_embeddings=image_contents[i]["embedding"],
                )
                ok_i += 1
            except Exception:
                bad_i += 1
        print(f"图像notes添加完成: {ok_i} 成功, {bad_i} 失败")
         
        # --- Initialize local links for text notes only ---
        text_count = ok
        for i, note in enumerate(list(ms.memories.values())[:text_count]):
            for d in range(-window_size, window_size + 1):
                if d == 0:
                    continue
                j = i + d
                if 0 <= j < text_count:
                    note.links.append(j)
            note.links = list(set(note.links))
        print("links initialized")
        
        
        
        # save the memory system before evolving
        print(f"\n保存Memory System: {name}")
        print(f"  准备保存 {len(ms.memories)} 条记忆...")
        ms.save_memory_system(name+"_iter_0")
         
        # --- Evolve (optional) ---
        for it in range(evolve_iters):
            print(f"Evolving memory system: iteration {it + 1}")
            evolution_start = time.time()
            _ = await ms.process_memory_all()
            evolution_duration = time.time() - evolution_start
            add_stage_duration("memory_evolution", evolution_duration)
            
            print(f"Re-embedding after evolution iteration {it + 1}")
            re_embedding_start = time.time()
            meta = [n.context + " keywords: " + ", ".join(n.keywords) for n in ms.memories.values()]
            pre = await embed_many(meta, kind="re_embedding")
            ms.reset_retriever()
            ms.retriever.add_documents(meta, pre)
            re_embedding_duration = time.time() - re_embedding_start
            add_stage_duration("re_embedding", re_embedding_duration)
            
            # exclude itself in links
            for i, note in enumerate(list(ms.memories.values())):
                note.links = [j for j in note.links if j != i]
                
            # save the memory system after each iteration
            print(f"\n保存Memory System: {name}")
            print(f"  准备保存 {len(ms.memories)} 条记忆...")
            ms.save_memory_system(name+"_iter_"+str(it+1))

    if not flag_dataset:
        for p in pdf_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("Memory System 构建完成！")
    print(f"   - 名称: {name}")
    print(f"   - 总记忆数: {len(ms.memories)}")
    print(f"   - 保存路径: {MEMORY_SYSTEMS_DIR}/{name}/")
    print("=" * 80 + "\n")
    return ms


def search_memory(memory_system: AgenticMemorySystem, query_or_keywords, k: int = 10, modality: str = "all", method: str = "semantic", top_k_text: int = 5, top_k_image: int = 5):
    if method == "semantic":
        _, notes = memory_system.find_related_notes_original(str(query_or_keywords), k=k, modality=modality)
        return notes
    elif method == "keywords":
        return memory_system.search_keyword(keywords=list(query_or_keywords), modality=modality, top_k_text=top_k_text, top_k_image=top_k_image)
    else:
        _, notes = memory_system.find_related_notes_original(str(query_or_keywords), k=k, modality=modality)
        return notes
