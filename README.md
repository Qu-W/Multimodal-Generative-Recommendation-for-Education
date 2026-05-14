# Multimodal-Generative-Recommendation-for-Education
A generative course recommendation system built on the [MOOCCube](http://moocdata.cn/data/MOOCCube) dataset, combining **semantic retrieval**, **collaborative filtering**, and **LLM reranking** into a unified pipeline. Supports both API-based LLMs (Qwen / OpenAI-compatible) and locally fine-tuned models via LoRA SFT.

---
# Features

- **Semantic Retrieval** — BGE-M3 embeddings + FAISS IndexFlatIP
- **Collaborative Filtering** — BPR trained on user interaction sequences
- **Hybrid Retrieval** — weighted fusion of semantic and CF scores
- **LLM Reranking** — prompt-based reranking with natural language explanations
- **LoRA SFT** — fine-tune a local LLM on domain-specific ranking data
- **CLI Interface** — interactive terminal demo, no web server required
- **MMRec Integration** — multimodal recommendation baselines (TMRec, BM3, LightGCN, etc.)

---
## Requirements

- Python 3.10+
- PyTorch 2.x + CUDA (recommended for LoRA training)
- ~10 GB VRAM for LoRA training with Qwen2.5-1.5B

conda create -n edurec python=3.10
conda activate edurec

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install faiss-cpu sentence-transformers transformers
pip install openai pyyaml pandas numpy

# For LoRA training only
pip install peft trl datasets

## Dataset

Download [MOOCCube](http://moocdata.cn/data/MOOCCube) and place under `data/raw/MOOCCube/`:

```
data/raw/MOOCCube/
├── entities/
│   ├── course.json
│   └── concept.json
├── relations/
│   ├── course-concept.json
│   └── user-course.json
└── ...
```

**Dataset statistics after preprocessing:**
| Metric | Value |
|--------|-------|
| Users | 34,857 |
| Courses | 706 |
| Interactions | 272,814 |
| Avg. interactions / user | 7.8 |
| Split strategy | Leave-one-out |
---

## Quick Start
### 1. Preprocess
```bash
python scripts/preprocess.py
```
Outputs to `data/processed/`: `courses.json`, `interactions.csv`, `user_sequences.json`, `train/valid/test.json`.

### 2. Build FAISS Index
```bash
python scripts/build_index.py
```
Encodes all courses with BGE-M3 and saves `data/index/course.faiss`.

### 3. Train BPR
```bash
python scripts/train_bpr.py --dim 64 --epochs 50
```
Saves `data/processed/bpr_embeddings.npz`.

### 4. Configure
Edit `configs/config.yaml`:
```yaml
embedding:
  model_name: "/path/to/bge-m3"     # local BGE-M3 model path

generation:
  provider: "qwen"                   # "qwen" (API) or "local" (LoRA)
  api_key: "sk-..."                  # DashScope API key
```

### 5. Run
```bash
# Interactive CLI (recommended)
python cli.py

# HTTP server
python server.py
# Open http://127.0.0.1:7860
```

---
## LoRA SFT Pipeline
Fine-tune a local LLM on MOOCCube-specific ranking data for improved reranking quality.
```bash
# 1. Build SFT training data from interaction sequences
python scripts/build_sft_data.py --max_users 5000 --neg_k 7
# → data/sft/train.jsonl, data/sft/valid.jsonl

# 2. Download base model
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/path/to/models')
"

# 3. Train
python scripts/train_lora.py \
    --base_model "/path/to/Qwen2.5-1.5B-Instruct" \
    --output_dir models/lora_reranker \
    --epochs 3 --batch 1 --grad_accum 16
# → models/lora_reranker/

# 4. Switch to local mode
# Edit configs/config.yaml:
#   generation:
#     provider: "local"
#     base_model: "/path/to/Qwen2.5-1.5B-Instruct"
#     lora_adapter: "models/lora_reranker"
```
---
## MMRec Baselines
Train multimodal recommendation baselines (TMRec, BM3, LightGCN, etc.) using the MMRec framework:
```bash
# Convert to MMRec format (requires faiss-cpu)
python scripts/convert_to_mmrec.py

# Train TMRec
cd MMRec-master/src
python main.py -m TMRec -d mooccube

# Train BM3
python main.py -m BM3 -d mooccube
```
---
