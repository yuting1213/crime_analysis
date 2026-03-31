# 5090 部署指南

## 1. Clone 專案

```bash
git clone https://github.com/yuting1213/crime_analysis.git
cd crime_analysis
```

## 2. 建立環境

```bash
conda create -n crime python=3.11 -y
conda activate crime

# PyTorch CUDA（5090 用 CUDA 12.8+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 核心依賴
pip install transformers Pillow scikit-learn opencv-python
pip install rank-bm25 jieba chromadb sentence-transformers
pip install google-genai  # LLM Judge (Gemini)

# 可選：DeepFace + MediaPipe（protobuf 衝突需注意）
# pip install deepface tf-keras mediapipe==0.10.14
# 注意：tensorflow 和 mediapipe 的 protobuf 版本衝突
# 建議先不裝，確認核心功能後再處理
```

## 3. SSL 修正（Anaconda 環境）

```bash
# conda 會設一個不存在的 SSL_CERT_FILE，影響 HuggingFace 下載
unset SSL_CERT_FILE
```

## 4. 下載模型

```bash
# ViT-Base（自動下載到 ~/.cache/huggingface）
python -c "from transformers import ViTModel, ViTImageProcessor; ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'); ViTModel.from_pretrained('google/vit-base-patch16-224')"

# BGE-M3（RAG embedding，約 2.2GB）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Qwen2.5-7B-Instruct（報告生成，約 15GB）— 5090 才跑得了
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto', device_map='auto')"
```

## 5. 準備資料

```bash
# UCF-Crime 影片放到 UCA/ 下（同 3060 的結構）
# UCA/UCF_Crimes/UCF_Crimes/Videos/{Category}/*.mp4

# 建立 RAG 索引
cd crime_analysis
python data/scripts/build_rag.py
```

## 6. 訓練 MIL Head

```bash
# 如果從 3060 帶了 feature_cache/v2/ 過來，放到 outputs/ 下可跳過特徵提取
# 否則會自動重新提取（約 20 分鐘）
python -m scripts.train_mil --epochs 30 --lr 1e-4 --batch_size 16
```

## 7. 驗證 Pipeline

```bash
# 快速測試 3 個樣本
python -m scripts.pilot_experiment --n_samples 3 --split Test

# 正式 Pilot（30 個樣本）
python -m scripts.pilot_experiment --n_samples 30 --split Test
```

## 8. 5090 專屬：Qwen2.5-7B 報告生成

3060 跑不了的功能，5090 可以：

```bash
# 確認 Qwen2.5 能載入
python -c "
from agents.planner import PlannerAgent
p = PlannerAgent.__new__(PlannerAgent)
p._report_model = None
p._report_tokenizer = None
p._load_report_model()
print('Qwen2.5 loaded successfully')
"
```

載入成功後，pipeline 的 Step 3b 會自動使用 Qwen2.5 生成報告（不再走 fallback）。

## 9. 5090 專屬：DPO 訓練

```bash
# 需要先設定 Gemini API key（用於 Judge）
export GEMINI_API_KEY="你的key"

# 生成 DPO 偏好對（每類 15 對，約 195 對）
python -c "
from pipeline import CrimeAnalysisPipeline
pipeline = CrimeAnalysisPipeline()
# 需要準備 video_samples，參考 pipeline.build_dpo_dataset()
"
```

## 10. 消融實驗

```bash
python -m scripts.run_ablation --n_samples 100 --split Test
```

## 從 3060 帶過來的檔案（可選）

```
outputs/
├── feature_cache/v2/     # 1165 個 .npy（省 ~20 分鐘重新提取）
├── mil_weights/          # 訓練好的權重（省 ~5 分鐘重新訓練）
└── pilot_summary.json    # 之前的 pilot 結果（參考用）
```

## 環境差異注意

| 項目 | 3060 (12GB) | 5090 (32GB) |
|------|-------------|-------------|
| Qwen2.5-7B | fallback | **fp16 可跑** |
| batch_size (MIL) | 16 | 可加到 **32-64** |
| DPO 訓練 | 不可 | **可以** |
| 特徵提取速度 | ~1165 影片/20min | 更快 |
