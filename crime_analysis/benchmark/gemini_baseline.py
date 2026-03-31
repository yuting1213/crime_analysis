"""
Gemini Baseline — 基準報告生成模組

作為 ablation 實驗的 baseline：直接把影片丟給 Gemini，
不經過 multi-agent / RAG / Reflector，生成鑑定報告。

用途：
  1. 比較 multi-agent 架構 vs 單模型直接生成的報告品質
  2. 驗證 H-RAG + Reflector 的實際貢獻
  3. 作為 LLM-as-Judge pairwise 比較的對照組

支援兩種輸入模式：
  1. 影片檔案直接上傳（Gemini 2.0 原生支援）
  2. 幀序列上傳（fallback，或用於和系統做相同輸入的公平比較）
"""
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import cfg
from rag.rag_module import LEGAL_ELEMENTS, GROUP_LEGAL_CONTEXT

logger = logging.getLogger(__name__)

# Gemini 報告生成 prompt（與 Step 3b 結構對齊，方便公平比較）
GEMINI_SYSTEM_PROMPT = """\
你是一位台灣刑事鑑定報告撰寫專家。根據提供的影片內容，撰寫一份結構化的初步鑑定報告。

報告要求：
1. 必須以繁體中文撰寫
2. 必須涵蓋所有適用的法律構成要件（逐一論述是否該當）
3. 引用具體法條條號（台灣刑法）
4. 區分「影片可觀察事實」與「推論」
5. 若有不確定性，須明確說明
"""

GEMINI_USER_TEMPLATE = """\
請分析這段影片，判斷是否涉及犯罪行為，並撰寫初步鑑定報告。

## 可能適用的犯罪類別
{categories}

## 輸出格式
請依下列結構撰寫報告：

### 一、事實認定
（根據影片觀察到的客觀事實，標註關鍵時間點）

### 二、犯罪類別判定
（判定最可能的犯罪類別，說明信心程度）

### 三、構成要件分析
（逐一論述每個構成要件是否該當，引用影片證據）

### 四、法律適用
（引用具體法條條號）

### 五、不確定性與限制
（影片分析的限制、無法確認的事項）

### 六、初步結論
（綜合判斷）
"""


class GeminiBaseline:
    """
    Gemini 基準報告生成器。

    使用方式：
        baseline = GeminiBaseline()

        # 模式 1：直接上傳影片
        report = baseline.generate_from_video("path/to/video.mp4")

        # 模式 2：幀序列
        report = baseline.generate_from_frames(frames, metadata)
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        """延遲初始化 Gemini client。"""
        if self._client is not None:
            return self._client
        try:
            import google.generativeai as genai
            if self._api_key:
                genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.model_name)
            logger.info(f"[GeminiBaseline] 初始化 {self.model_name}")
        except ImportError:
            raise ImportError(
                "需要安裝 google-generativeai：pip install google-generativeai"
            )
        return self._client

    # ── 模式 1：影片檔案上傳 ─────────────────────────────────

    def generate_from_video(
        self,
        video_path: str,
        crime_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        直接上傳影片到 Gemini API 生成報告。

        Gemini 2.0 支援影片上傳（最大 2GB / 1hr），
        API 會自動抽幀和理解影片內容。
        """
        import google.generativeai as genai

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"影片不存在：{video_path}")

        client = self._get_client()

        # 上傳影片
        logger.info(f"[GeminiBaseline] 上傳影片：{path.name}")
        video_file = genai.upload_file(str(path))

        # 等待處理完成
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"影片處理失敗：{video_file.state.name}")

        # 組裝 prompt
        prompt = self._build_prompt(crime_type_hint)

        # 生成
        response = client.generate_content(
            [video_file, prompt],
            generation_config={"temperature": 0.7, "max_output_tokens": 2048},
        )

        report_text = response.text
        logger.info(f"[GeminiBaseline] 影片報告生成完成（{len(report_text)} chars）")

        # 清理上傳的檔案
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

        return self._wrap_result(report_text, video_path, crime_type_hint)

    # ── 模式 2：幀序列輸入 ───────────────────────────────────

    def generate_from_frames(
        self,
        frames: List,
        video_metadata: Optional[Dict] = None,
        crime_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        將幀序列轉為圖片後送入 Gemini 生成報告。
        用於和系統使用相同輸入的公平比較。

        均勻抽取最多 8 幀（Gemini 圖片上限較影片低）。
        """
        try:
            import PIL.Image
        except ImportError:
            raise ImportError("需要安裝 Pillow：pip install Pillow")

        client = self._get_client()

        # 均勻抽取最多 8 幀
        max_frames = 8
        valid_frames = [f for f in frames if f is not None]
        if not valid_frames:
            return self._wrap_result("（無有效幀可供分析）", "frames", crime_type_hint)

        step = max(1, len(valid_frames) // max_frames)
        selected = valid_frames[::step][:max_frames]

        # 轉為 PIL Image
        images = []
        for frame in selected:
            if hasattr(frame, "shape"):  # numpy array (OpenCV BGR)
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(PIL.Image.fromarray(rgb))
            elif isinstance(frame, PIL.Image.Image):
                images.append(frame)

        if not images:
            return self._wrap_result("（無法轉換幀為圖片）", "frames", crime_type_hint)

        # 組裝 prompt
        prompt = self._build_prompt(crime_type_hint)
        frame_desc = f"以下是影片中均勻抽取的 {len(images)} 幀截圖，請根據這些畫面進行分析。\n\n"

        # Gemini multimodal: [image1, image2, ..., text]
        content_parts = list(images) + [frame_desc + prompt]

        response = client.generate_content(
            content_parts,
            generation_config={"temperature": 0.7, "max_output_tokens": 2048},
        )

        report_text = response.text
        video_id = video_metadata.get("video_id", "unknown") if video_metadata else "unknown"
        logger.info(f"[GeminiBaseline] 幀報告生成完成（{len(report_text)} chars, {len(images)} frames）")

        return self._wrap_result(report_text, video_id, crime_type_hint)

    # ── 批次生成 ─────────────────────────────────────────────

    def batch_generate(
        self,
        samples: List[Dict],
        mode: str = "video",
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批次生成 baseline 報告。

        Args:
            samples: [{"video_path": str, "ground_truth": str, ...}, ...]
            mode: "video"（直接上傳）或 "frames"（幀序列）
            output_path: 若提供，自動存為 JSONL
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"[GeminiBaseline] {i+1}/{len(samples)}: {sample.get('video_id', '?')}")
            try:
                if mode == "video" and "video_path" in sample:
                    result = self.generate_from_video(
                        sample["video_path"],
                        crime_type_hint=sample.get("ground_truth"),
                    )
                elif "frames" in sample:
                    result = self.generate_from_frames(
                        sample["frames"],
                        video_metadata=sample.get("metadata"),
                        crime_type_hint=sample.get("ground_truth"),
                    )
                else:
                    logger.warning(f"樣本 {i} 無有效輸入，跳過")
                    continue

                result["ground_truth"] = sample.get("ground_truth", "")
                result["video_id"] = sample.get("video_id", str(i))
                results.append(result)

            except Exception as e:
                logger.error(f"[GeminiBaseline] 樣本 {i} 失敗：{e}")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"[GeminiBaseline] {len(results)} 份報告 → {output_path}")

        return results

    # ── 內部方法 ─────────────────────────────────────────────

    def _build_prompt(self, crime_type_hint: Optional[str] = None) -> str:
        """組裝 Gemini prompt。若有 crime_type_hint 則提供對應法條參考。"""
        categories = ", ".join(
            c for c in cfg.data.crime_categories if c != "Normal"
        )

        base_prompt = GEMINI_SYSTEM_PROMPT + "\n\n" + GEMINI_USER_TEMPLATE.format(
            categories=categories
        )

        # 若有提示類別，附上法律構成要件作為參考
        if crime_type_hint and crime_type_hint in LEGAL_ELEMENTS:
            elements = LEGAL_ELEMENTS[crime_type_hint]
            articles = GROUP_LEGAL_CONTEXT.get(crime_type_hint, [])
            base_prompt += f"\n\n## 參考（非必須遵循）\n"
            base_prompt += f"- 提示犯罪類別：{crime_type_hint}\n"
            base_prompt += f"- 可能適用法條：{'、'.join(articles)}\n"
            base_prompt += f"- 構成要件：{'、'.join(elements)}\n"

        return base_prompt

    def _wrap_result(
        self,
        report_text: str,
        source: str,
        crime_type_hint: Optional[str],
    ) -> Dict[str, Any]:
        """包裝生成結果為統一格式。"""
        return {
            "source": "gemini_baseline",
            "model": self.model_name,
            "report_text": report_text,
            "input_source": source,
            "crime_type_hint": crime_type_hint,
        }
