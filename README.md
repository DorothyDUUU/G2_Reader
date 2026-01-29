# 💻 G2‑Reader: Dynamic DAG‑based Document Reader for Multi‑modal Long‑Document Understanding

[![project](https://img.shields.io/badge/project-Page-blue)](#)
[![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b.svg)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> 🎯 **D2‑Reader** (Dynamic DAG-based Reader) is an intelligent agent system specialized for **multi‑modal long‑document understanding**. It dynamically decomposes complex document-based queries into a **Directed Acyclic Graph (DAG)** of sub-tasks and utilizes an **Agentic Memory System (AMS)** to efficiently retrieve and reason over text, figures, and tables in long PDF documents.

---

## ✨ Highlights

* 🏗️ **Dynamic DAG Decomposition** – Automatically breaks down global complex queries into structured sub-tasks with logical dependencies.
* 🧠 **Agentic Memory System (AMS)** – A specialized document memory layer that provides unified indexing and retrieval for multi-modal contents (text, images, and tables).
* 🔍 **Hybrid Retrieval Strategy** – Combines Semantic Search with Keyword-based (BM25) retrieval, enhanced by visual feature matching for high recall.
* 🤖 **Self-Refinement Loop** – Built-in evidence sufficiency checker that dynamically adjusts the DAG structure to fill information gaps during reasoning.
* ⚙️ **Production-Ready Architecture** – Features process-safe execution counters, comprehensive token usage tracking, and multi-process concurrency support.

---

## 🚀 Getting Started

### 1. 🛠️ Installation

```bash
# Create environment
conda create -n d2-reader python=3.10
conda activate d2-reader

# Clone repository
git clone https://github.com/justLittleWhite/D2-Reader.git
cd D2-Reader

# Install dependencies
pip install -r requirements.txt
```

### 2. ⚙️ Configuration

Set up your LLM API and base paths in `config/config.py`:

```python
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "your-api-key"

# Data root directory
DATA_ROOT = "/path/to/your/data"
```

### 3. ⏱️ Quick Inference

You can run full evaluations or single inference tasks using the provided scripts.

**Run batch evaluation:**
```bash
bash scripts/D2reader.sh
```

**Run single inference task:**
```bash
python -m scripts.test_rag \
    --data_path "data/processed_sample.jsonl" \
    --save_dir "results/output" \
    --model "qwen3-vl-32b-instruct" \
    --use_dag
```

---

## 📁 Project Structure

```text
D2-Reader/
├── agent_search/      # Core Logic: DAG decomposition, reasoners, and execution engine
├── prebuild/          # Preprocessing: PDF parsing (MinerU/OCR) and Memory construction
├── config/            # Configuration: Model parameters and Prompt templates
├── scripts/           # Execution: Batch evaluation, inference, and sample building
├── data/              # Data storage (Excluded from git)
└── results/           # Output logs and inference results
```

---

## 📢 News
* [2026-01-28] 🎉 Project cleanup completed and code successfully uploaded to GitHub.
* [2026-01-xx] 📄 D2-Reader paper submitted.

---

## ✍️ Citation

If you find D2-Reader useful for your research, please cite:

```bibtex
@article{zhou2026d2reader,
  title={D2-Reader: Dynamic DAG-based Document Reader for Multi-modal Long-Document Understanding},
  author={Yifan Zhou and et al.},
  journal={arXiv preprint arXiv:25xx.xxxxx},
  year={2026}
}
```

---

## 📝 License

This project is licensed under the **Apache 2.0** License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
We thank **MinerU** for providing high-quality PDF parsing capabilities, and the open-source community for the foundational models (OpenAI/Qwen) that power this project.
