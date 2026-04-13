#!/bin/bash
# VLM 分類能力 benchmark — 依序測試多個模型
# 用法: bash scripts/run_vlm_benchmark.sh

set -e
cd "$(dirname "$0")/.."
unset SSL_CERT_FILE
export TF_CPP_MIN_LOG_LEVEL=3
PYTHON=$HOME/miniconda3/envs/crime/bin/python

echo "=========================================="
echo "VLM Classification Benchmark"
echo "=========================================="

# 1. Gemma-4-26B-A4B (正在跑，跳過如果結果已存在)
if [ ! -f outputs/vlm_diagnostic_gemma4_26b.json ]; then
    echo "[1/3] Gemma-4-26B-A4B — running..."
    # 已經在背景跑了
else
    echo "[1/3] Gemma-4-26B-A4B — already done, skipping"
fi

# 2. Gemma-4-31B
echo "[2/3] Gemma-4-31B-it (INT4)..."
$PYTHON -c "
import random, sys, json, torch, re, cv2
from pathlib import Path
from PIL import Image
from collections import Counter
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, '.')
from scripts.pilot_experiment import load_pilot_samples, CRIME_CATEGORIES

random.seed(42)
samples = load_pilot_samples(n_samples=52, split='Test')

print('Loading Gemma-4-31B INT4...')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
model = AutoModelForImageTextToText.from_pretrained('google/gemma-4-31B-it', quantization_config=bnb_config, device_map='auto')
processor = AutoProcessor.from_pretrained('google/gemma-4-31B-it')
print('Model ready!')

PROMPT = 'You are a forensic surveillance video analyst.\nLook at these frames from a CCTV video and determine what crime is occurring.\n\nChoose ONE category from: ' + ', '.join(CRIME_CATEGORIES) + '\n\nCategory definitions:\n- Assault: One person attacking another (one-sided)\n- Robbery: Forcibly taking belongings\n- Stealing: Secretly taking items\n- Shoplifting: Concealing store merchandise\n- Burglary: Breaking into a building/car\n- Fighting: Mutual physical combat (both sides)\n- Arson: Deliberately setting fire\n- Explosion: Sudden blast with smoke/debris\n- RoadAccidents: Vehicle collision\n- Vandalism: Deliberately damaging property\n- Abuse: Sustained harm to vulnerable person\n- Shooting: Gunfire, weapon visible\n- Arrest: Law enforcement restraining suspect\n\nReply with ONLY: CATEGORY: <name>'

def extract_frames(path, n=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def parse_cat(resp):
    m = re.search(r'CATEGORY:\s*(\w+)', resp, re.IGNORECASE)
    if m:
        for c in CRIME_CATEGORIES:
            if c.lower() == m.group(1).lower(): return c
    for c in CRIME_CATEGORIES:
        if c.lower() in resp.lower(): return c
    return 'Normal'

results = []
for i, s in enumerate(samples):
    gt = s['ground_truth']
    frames = extract_frames(s['video_path'])
    messages = [{'role': 'user', 'content': [*[{'type': 'image', 'image': img} for img in frames], {'type': 'text', 'text': PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    resp = processor.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    pred = parse_cat(resp)
    ok = pred == gt
    results.append({'video_id': s['video_id'], 'ground_truth': gt, 'predicted': pred, 'correct': ok})
    print(f'[{i+1}/52] {s[\"video_id\"]:25s} GT={gt:15s} -> {pred:15s} {\"V\" if ok else \"X\"}')

c = sum(1 for r in results if r['correct'])
print(f'\nGemma-4-31B INT4: {c}/52 ({100*c/52:.1f}%)')
with open('outputs/vlm_diagnostic_gemma4_31b.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
" 2>&1 | tee -a vlm_benchmark.log

# 清 VRAM
echo "Clearing VRAM..."
sleep 5

# 3. Qwen3.5-9B
echo "[3/3] Qwen3.5-9B..."
$PYTHON -c "
import random, sys, json, torch, re, cv2, gc
from pathlib import Path
from PIL import Image
from collections import Counter
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, '.')
from scripts.pilot_experiment import load_pilot_samples, CRIME_CATEGORIES

random.seed(42)
samples = load_pilot_samples(n_samples=52, split='Test')

print('Loading Qwen3.5-9B...')
model = AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3.5-9B', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
print('Model ready!')

PROMPT = 'You are a forensic surveillance video analyst.\nLook at these frames from a CCTV video and determine what crime is occurring.\n\nChoose ONE category from: ' + ', '.join(CRIME_CATEGORIES) + '\n\nCategory definitions:\n- Assault: One person attacking another (one-sided)\n- Robbery: Forcibly taking belongings\n- Stealing: Secretly taking items\n- Shoplifting: Concealing store merchandise\n- Burglary: Breaking into a building/car\n- Fighting: Mutual physical combat (both sides)\n- Arson: Deliberately setting fire\n- Explosion: Sudden blast with smoke/debris\n- RoadAccidents: Vehicle collision\n- Vandalism: Deliberately damaging property\n- Abuse: Sustained harm to vulnerable person\n- Shooting: Gunfire, weapon visible\n- Arrest: Law enforcement restraining suspect\n\nReply with ONLY: CATEGORY: <name>'

def extract_frames(path, n=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def parse_cat(resp):
    m = re.search(r'CATEGORY:\s*(\w+)', resp, re.IGNORECASE)
    if m:
        for c in CRIME_CATEGORIES:
            if c.lower() == m.group(1).lower(): return c
    for c in CRIME_CATEGORIES:
        if c.lower() in resp.lower(): return c
    return 'Normal'

results = []
for i, s in enumerate(samples):
    gt = s['ground_truth']
    frames = extract_frames(s['video_path'])
    messages = [{'role': 'user', 'content': [*[{'type': 'image', 'image': img} for img in frames], {'type': 'text', 'text': PROMPT}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    resp = processor.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    pred = parse_cat(resp)
    ok = pred == gt
    results.append({'video_id': s['video_id'], 'ground_truth': gt, 'predicted': pred, 'correct': ok})
    print(f'[{i+1}/52] {s[\"video_id\"]:25s} GT={gt:15s} -> {pred:15s} {\"V\" if ok else \"X\"}')

c = sum(1 for r in results if r['correct'])
print(f'\nQwen3.5-9B: {c}/52 ({100*c/52:.1f}%)')
with open('outputs/vlm_diagnostic_qwen35_9b.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
" 2>&1 | tee -a vlm_benchmark.log

echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
