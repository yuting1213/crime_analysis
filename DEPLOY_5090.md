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
pip install google-genai matplotlib  # LLM Judge (Gemini, 僅 DPO 時需要) + 訓練視覺化

# MediaPipe + DeepFace（4/4 模態特徵提取）
pip install mediapipe==0.10.14
pip install deepface tf-keras
# protobuf 衝突修復：降回 tensorflow 2.16 + jax 0.4.30
pip install "tensorflow==2.16.*" "tf-keras==2.16.*" protobuf==4.25.9
pip install "jax==0.4.30" "jaxlib==0.4.30"

# torch.compile 需要 C compiler（conda 環境需額外安裝）
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
```

## 3. SSL 修正（Anaconda 環境）

```bash
# conda 會設一個不存在的 SSL_CERT_FILE，影響 HuggingFace 下載
unset SSL_CERT_FILE
```

## 4. 下載模型（全部本地執行，不需要 API key）

```bash
# ViT-Base（自動下載到 ~/.cache/huggingface，約 340MB）
python -c "from transformers import ViTModel, ViTImageProcessor; ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'); ViTModel.from_pretrained('google/vit-base-patch16-224')"

# BGE-M3（RAG embedding，約 2.2GB）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Qwen3-VL-8B-Instruct（統一 VLM：分類+報告生成，BF16 約 16GB）
python -c "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct'); Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-8B-Instruct', torch_dtype='auto', device_map='auto')"
```

> **所有模型都是開源的，下載到本地後完全離線運作，不消耗任何 API token。**
> HuggingFace 可能提示設定 `HF_TOKEN`，那只是加速下載用，不設也能下載。

## 5. 準備資料

```bash
# UCF-Crime 影片結構：
# UCA/UCF_Crimes/UCF_Crimes/Videos/{Category}/*.mp4     # 950 anomaly
# UCA/UCF_Crimes/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/*.mp4  # 800 normal
# UCA/UCF_Crimes/UCF_Crimes/Videos/z_Normal_Videos_event/*.mp4           # 50 normal

# Normal 影片需另外下載：
#   Training-Normal-Videos-Part-1.zip (~36GB, 430 支)
#   Training-Normal-Videos-Part-2.zip (~32GB, 370 支)

# 建立 RAG 索引
cd crime_analysis
python data/scripts/build_rag.py
```

## 6. 訓練 MIL Head（預篩分類器）

```bash
# 超參數已寫入 config.py TrainingConfig
# 首次會提取 4 模態特徵 → outputs/feature_cache/v4/（之後自動快取）
python -m scripts.train_mil --split Train

# 產出：
#   outputs/mil_weights/fusion_encoder.pt
#   outputs/mil_weights/crime_head.pt
#   outputs/mil_weights/escalation_head.pt
#   outputs/mil_weights/feature_scaler.npz
#   outputs/mil_weights/training_curves.png
```

## 7. 驗證 Pipeline

```bash
# 快速測試 3 個樣本
python -m scripts.pilot_experiment --n_samples 3 --split Test

# 正式 Pilot（13 個樣本，每類一支）
python -m scripts.pilot_experiment --n_samples 13 --split Test

# 報告輸出位置：outputs/pilot_reports/{video_id}.txt
```

## 8. 5090 專屬：Qwen3-VL 視覺分類+報告

5090 才能跑的功能（需要 ~16GB VRAM）：

```bash
# 確認 Qwen3-VL 能載入
python -c "
from agents.planner import PlannerAgent
p = PlannerAgent.__new__(PlannerAgent)
p._report_model = None
p._report_tokenizer = None
p._load_report_model()
print('Qwen3-VL-8B loaded successfully')
"
```

Qwen3-VL 統一負責：
- **Step 2b**：看 4 張關鍵幀 → zero-shot 犯罪分類（覆核 MIL Head）
- **Step 3b**：帶幀生成繁體中文鑑定報告（含法律構成要件分析）

## 9. DPO 訓練（可選，需要 Gemini API key）

```bash
# 唯一需要 API key 的功能（用 Gemini 做偏好對比較）
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

## 模型清單（全部本地運行）

| 模型 | 用途 | VRAM | API Key |
|------|------|------|---------|
| Qwen3-VL-8B-Instruct | 分類 + 報告 | ~16GB | 不需要 |
| R3D-18 (torchvision) | 行為特徵提取 | ~45MB | 不需要 |
| ViT-Base | 視覺語意特徵 | ~340MB | 不需要 |
| BGE-M3 | RAG 向量檢索 | ~2.2GB | 不需要 |
| MediaPipe Pose | 姿態特徵 | CPU only | 不需要 |
| DeepFace | 情緒特徵 | CPU only | 不需要 |
| Gemini 2.0 Flash | DPO Judge | Cloud | `GEMINI_API_KEY`（僅 DPO） |

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
| class_weights | 自動計算 | Explosion 6.3x, RoadAccidents 0.4x |
| feature_version | v4 | R3D(無L2norm) + ViT + Pose + Emotion + StandardScaler |

## 環境差異注意

| 項目 | 3060 (12GB) | 5090 (32GB) |
|------|-------------|-------------|
| Qwen3-VL-8B | 無法載入 | **BF16 可跑（~16GB）** |
| MIL Head 分類 | 僅 MIL Head | **MIL + VLM 覆核** |
| 報告生成 | fallback（結構化模板） | **Qwen3-VL 帶幀生成** |
| batch_size (MIL) | 16 | **32** |
| DPO 訓練 | 不可 | **可以** |
| 4/4 模態推理 | 可能 OOM | **充裕** |
