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

# PyTorch CUDA（5090 Blackwell sm_120 需要 cu128）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 核心依賴
pip install transformers accelerate Pillow scikit-learn opencv-python
pip install rank-bm25 jieba chromadb sentence-transformers
pip install google-genai matplotlib  # LLM Judge (Gemini) + 訓練視覺化

# MediaPipe + DeepFace（4/4 模態特徵提取）
# 安裝順序很重要：先裝 mediapipe，再裝 deepface + tensorflow
pip install mediapipe==0.10.14
pip install deepface tf-keras
# protobuf 衝突修復：降回 tensorflow 2.16 + jax 0.4.30
pip install "tensorflow==2.16.*" "tf-keras==2.16.*" protobuf==4.25.9
pip install "jax==0.4.30" "jaxlib==0.4.30"

# torch.compile 需要 C compiler（conda 環境需額外安裝）
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y

# Flash Attention 2（可選，目前 flash-attn 尚不支援 sm_120 Blackwell）
# pip install flash-attn --no-build-isolation
# 不裝也沒關係，planner.py 會自動 fallback 到預設注意力
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

# Qwen3-8B（報告生成，BF16 約 16GB）— 5090 才跑得了
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype='auto', device_map='auto')"
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
# 超參數已寫入 config.py TrainingConfig（bs=32, ep=60, lr=5e-4, λ_MIL=0.3）
# 不帶引數會自動讀取 config 預設值
# 首次會提取 R3D-18 + ViT 特徵 → outputs/feature_cache/v2/（之後自動快取）
python -m scripts.train_mil --split Train

# 產出：outputs/mil_weights/ (fusion_encoder.pt, crime_head.pt, escalation_head.pt)
# 訓練曲線圖：outputs/mil_weights/training_curves.png
```

## 7. 驗證 Pipeline

```bash
# 快速測試 3 個樣本
python -m scripts.pilot_experiment --n_samples 3 --split Test

# 正式 Pilot（30 個樣本）
python -m scripts.pilot_experiment --n_samples 30 --split Test
```

## 8. 5090 專屬：Qwen3-8B 報告生成

3060 跑不了的功能，5090 可以：

```bash
# 確認 Qwen3-8B 能載入
python -c "
from agents.planner import PlannerAgent
p = PlannerAgent.__new__(PlannerAgent)
p._report_model = None
p._report_tokenizer = None
p._load_report_model()
print('Qwen3-8B loaded successfully')
"
```

載入成功後，pipeline 的 Step 3b 會自動使用 Qwen3-8B 生成報告（non-thinking 模式，不再走 fallback）。

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

## 已驗證環境

以下為 2026-04-04 實際部署驗證通過的環境：

```
GPU:       NVIDIA GeForce RTX 5090 (32GB, sm_120 Blackwell)
Driver:    576.88 (CUDA 12.9)
PyTorch:   2.11.0+cu128
Python:    3.11.15
OS:        WSL2 (Ubuntu)
```

## 訓練參數（已調優）

| 參數 | 值 | 說明 |
|------|------|------|
| batch_size | 32 | 18 batch/epoch × 60 epoch = 1080 steps |
| epochs | 60 | 配合 cosine schedule 完整衰減 |
| lr | 5e-4 | + 5% warmup steps |
| lambda_mil | 0.3 | 主任務分類，MIL 輔助 |
| weight_decay | 1e-2 | 強正則化（13.3M params vs ~1K samples） |
| dropout | 0.2 | FusionEncoder Transformer dropout |
| label_smoothing | 0.1 | 緩解類別不平衡 |

## 資料需求

```
UCA/
├── UCFCrime_{Train|Test|Val}.json    # UCA 時序文字標註
├── UCF_Crimes/UCF_Crimes/Videos/
│   ├── {Abuse,Arrest,...,Vandalism}/  # 950 anomaly mp4
│   ├── Training_Normal_Videos_Anomaly/  # 800 normal mp4（需另外下載）
│   └── z_Normal_Videos_event/           # 50 normal mp4（UCF.zip 內含）
```

Normal 影片需從 UCF-Crime 官網另外下載：
- `Training-Normal-Videos-Part-1.zip` (~36GB, 430 支)
- `Training-Normal-Videos-Part-2.zip` (~32GB, 370 支)

## 環境差異注意

| 項目 | 3060 (12GB) | 5090 (32GB) |
|------|-------------|-------------|
| Qwen3-8B 報告生成 | fallback | **BF16 可跑（~16GB）** |
| batch_size (MIL) | 16 | **32**（更多 gradient steps） |
| Flash Attention 2 | 不支援 | **sm_120 待支援**（自動 fallback） |
| DPO 訓練 | 不可 | **可以** |
| 4/4 模態推理 | 可能 OOM | **充裕** |
| 特徵提取速度 | ~1085 影片/20min | **~3 分鐘（BF16 autocast）** |
