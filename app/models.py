from __future__ import annotations

import base64
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np

from .schemas import LocalModelConfig, VLMModelConfig
from .storage import ROOT


HAND_LABELS = {"hand", "hands", "left_hand", "right_hand", "person"}
OPEN_VOCAB_HAND_CLASSES = ["hand", "robot hand", "human hand", "robot gripper", "gripper"]
_OPEN_VOCAB_HAND_KEYS = {value.casefold() for value in OPEN_VOCAB_HAND_CLASSES}
# Open-vocabulary segmentation needs a larger canvas and a lower proposal
# threshold than the ordinary closed-set detector.  The fallback is only
# used when one side of the hand/object pair is still missing.
OPEN_VOCAB_DEFAULT_IMAGE_SIZE = 960
OPEN_VOCAB_FALLBACK_IMAGE_SIZE = 1280
OPEN_VOCAB_DEFAULT_CONFIDENCE = 0.12
_OPEN_VOCAB_CLASS_ALIASES = {
    "block": ("jenga block", "building block"),
    "grape": ("grapes",),
}

_ENGLISH_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every",
}
_ENGLISH_QUANTITIES = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first",
    "second", "third", "several", "many", "multiple", "few", "pair", "single", "double",
}
_ENGLISH_COLORS = {
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "purple", "pink",
    "brown", "orange", "peach", "lime", "olive", "violet", "cyan", "magenta", "beige", "gold", "golden", "silver", "transparent", "clear",
    "light", "dark", "colored", "colourful", "colorful",
}
_ENGLISH_MATERIALS = {
    "plastic", "wood", "wooden", "metal", "metallic", "steel", "aluminum", "aluminium", "glass",
    "ceramic", "paper", "cardboard", "rubber", "silicone", "fabric", "cloth", "leather", "foam",
}
_ENGLISH_SIZE_SHAPE = {
    "small", "smaller", "smallest", "large", "larger", "largest", "big", "tiny", "mini", "long",
    "short", "tall", "wide", "narrow", "thin", "thick", "round", "square", "rectangular", "circular",
}
# Product/type words such as "jenga" and "lego" carry useful visual class
# semantics; only generic styling words are stripped.
_ENGLISH_STYLE_MODIFIERS = {"toy", "style", "styled", "decorative"}
_ENGLISH_QUALITY_MODIFIERS = {"bright", "dim", "shiny", "matte", "clean", "dirty", "new", "old", "empty", "full"}
_AMBIGUOUS_COLOR_OBJECTS = {"orange", "peach", "lime", "olive"}
_ENGLISH_POSITION = {
    "left", "right", "upper", "lower", "top", "bottom", "front", "rear", "back", "center", "central",
    "middle", "nearby", "inner", "outer", "nearest", "farthest",
}
_ENGLISH_ACTION_MODIFIERS = {
    "pick", "picked", "picking", "grasp", "grasped", "grasping", "hold", "held", "holding", "move",
    "moved", "moving", "place", "placed", "placing", "put", "push", "pushed", "pushing", "pull",
    "pulled", "pulling", "open", "opened", "opening", "close", "closed", "closing", "remove", "removed",
    "removing", "insert", "inserted", "inserting", "lift", "lifted", "lifting", "target", "visible",
}
_ENGLISH_RELATION_WORDS = {
    "near", "beside", "alongside", "above", "below", "under", "behind", "ahead", "around", "between",
    "inside", "outside", "adjacent", "next", "from", "toward", "towards", "against", "with", "without",
}
_ENGLISH_GENERIC_NOUNS = {
    "object", "objects", "item", "items", "thing", "things", "target", "scene", "background", "area",
    "region", "location", "surface", "work", "workspace", "color", "colour", "material",
}
_ENGLISH_RELATION_TAIL = re.compile(
    r"\b(?:on|in|at|of|near|beside|alongside|above|below|under|behind|inside|outside|between|against|with|without|from|toward|towards|being|held\s+by|next\s+to|to\s+the\s+(?:left|right)(?:\s+of)?)\b",
    re.IGNORECASE,
)

_CHINESE_GENERIC_TERMS = {
    "物体", "物品", "东西", "目标", "目标物体", "场景", "背景", "区域", "位置", "颜色", "材质", "表面",
}
_CHINESE_MODIFIER_PATTERN = re.compile(
    r"^(?:(?:这|那|该|此|某)(?:一)?(?:个|只|块|件|张|把|根|瓶|杯)?|"
    r"(?:第)?[0-9一二两三四五六七八九十几多若干]+(?:个|只|块|件|张|把|根|瓶|杯|组|对)?|"
    r"(?:很|较|比较)?(?:小型|大型|小|大|迷你|长|短|高|矮|宽|窄|薄|厚|圆形|方形|矩形)的?|"
    r"(?:浅|深)?(?:红色|蓝色|绿色|黄色|黑色|白色|灰色|紫色|粉色|棕色|橙色|青色|米色|金色|银色|透明|彩色)的?|"
    r"(?:塑料制?|木制|木质|金属制?|钢制|铝制|玻璃制?|陶瓷制?|纸质|橡胶制?|硅胶制?|布制|皮质)的?|"
    r"(?:主要|可见|待处理|待操作|待抓取|被抓取|被拿起|被移动|正在抓取|正在拿起|抓取|拿起|移动|放置|操作|目标|玩具|乐高|叠叠乐)的?|"
    r"(?:左侧|右侧|前方|后方|上方|下方|中间|附近)的?)+"
)
_CHINESE_RELATION_MARKER = re.compile(r"(?:上面?|下面?|里面?|中|旁边|附近|左侧|右侧|前面|后面|上方|下方)的")
_CHINESE_RELATION_TAIL = re.compile(r"(?:位于|在|靠近|紧邻|处于|朝向)(?:桌面|桌子|容器|盒子|左侧|右侧|前方|后方|上方|下方|旁边|附近).*$")
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _split_object_phrases(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [
        item.strip()
        for item in re.split(r"\s+(?:and|or)\s+|[,;/|\n\r，；、]+", text, flags=re.IGNORECASE)
        if item.strip()
    ]


def _singularize_english_head(word: str) -> str:
    irregular = {
        "buses": "bus", "classes": "class", "knives": "knife", "leaves": "leaf", "shelves": "shelf",
    }
    if word in irregular:
        return irregular[word]
    if word in {"glasses", "scissors", "pliers", "tongs", "tweezers"}:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith(("ches", "shes", "xes", "zes")) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _normalize_english_object_phrase(value: str) -> str:
    text = value.casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"^(?:(?:near|beside|alongside|above|below|under|behind|next\s+to)\s+(?:the\s+)?)", "", text)
    text = re.sub(r"^(?:(?:a|an|one|two|several)\s+)?(?:pair|set|group|couple|piece|pieces|stack)\s+of\s+(?:the\s+)?", "", text)
    tail = _ENGLISH_RELATION_TAIL.search(text)
    if tail:
        text = text[:tail.start()]
    text = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", " ", text)
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)
    if len(tokens) == 1 and tokens[0] in _AMBIGUOUS_COLOR_OBJECTS:
        return tokens[0]
    removable = (
        _ENGLISH_DETERMINERS | _ENGLISH_QUANTITIES | _ENGLISH_COLORS | _ENGLISH_MATERIALS
        | _ENGLISH_SIZE_SHAPE | _ENGLISH_STYLE_MODIFIERS | _ENGLISH_QUALITY_MODIFIERS
        | _ENGLISH_POSITION | _ENGLISH_ACTION_MODIFIERS | _ENGLISH_RELATION_WORDS | {"up", "down", "to", "be"}
    )
    tokens = [
        token for token in tokens
        if token not in removable and not token.isdigit() and not re.fullmatch(r"\d+(?:st|nd|rd|th)", token)
    ]
    while tokens and tokens[-1] in _ENGLISH_GENERIC_NOUNS:
        tokens.pop()
    if not tokens:
        return ""
    tokens[-1] = _singularize_english_head(tokens[-1])
    phrase = " ".join(tokens).strip()
    if re.fullmatch(r"(?:left |right )?hands?", phrase):
        return "hand"
    if re.fullmatch(r"(?:human |person |gloved )(?:hands?|arms?)", phrase):
        return "human hand"
    if re.fullmatch(r"(?:robot|robotic|mechanical) (?:hands?|arms?)", phrase):
        return "robot hand"
    if re.fullmatch(r"(?:robot|robotic|mechanical) grippers?", phrase):
        return "robot gripper"
    return "" if phrase in _ENGLISH_GENERIC_NOUNS else phrase


def _normalize_chinese_object_phrase(value: str) -> str:
    text = re.sub(r"[\s\u3000]+", "", value)
    text = text.strip("的,，。.;；:：()（）[]【】")
    text = re.sub(r"^(?:被[^的]{1,16}(?:拿着|抓住|握住|移动)的|(?:放在|置于|位于)[^的]{1,16}的)", "", text)
    markers = list(_CHINESE_RELATION_MARKER.finditer(text))
    if markers and markers[-1].end() < len(text):
        text = text[markers[-1].end():]
    text = _CHINESE_RELATION_TAIL.sub("", text)
    previous = None
    while text and text != previous:
        previous = text
        text = _CHINESE_MODIFIER_PATTERN.sub("", text).lstrip("的")
    text = re.sub(r"们$", "", text).strip("的,，。.;；:：")
    if text in {"左手", "右手", "手", "双手"}:
        return "hand"
    if text in {"人手", "戴手套的手", "戴手套手"}:
        return "human hand"
    if text in {"机械臂", "机械手", "机器人手", "机器人手臂", "机器人机械臂"}:
        return "robot hand"
    if text in {"夹爪", "机械夹爪", "机器人夹爪"}:
        return "robot gripper"
    return "" if not text or text in _CHINESE_GENERIC_TERMS else text


def normalize_yoloe_object_terms(object_terms: list[Any]) -> list[str]:
    """Derive detectable YOLOE nouns without mutating VLM target terms."""
    output: list[str] = []
    seen: set[str] = set()
    for value in object_terms:
        for phrase in _split_object_phrases(value):
            normalized = _normalize_chinese_object_phrase(phrase) if re.search(r"[\u3400-\u9fff]", phrase) else _normalize_english_object_phrase(phrase)
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            output.append(normalized)
    return output


def _open_vocab_proximity_classes(object_classes: list[str]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    normalized_objects = [value for value in normalize_yoloe_object_terms(object_classes) if value.casefold() not in _OPEN_VOCAB_HAND_KEYS]
    expanded_objects: list[str] = []
    for value in normalized_objects:
        expanded_objects.extend(_OPEN_VOCAB_CLASS_ALIASES.get(value.casefold(), ()))
        expanded_objects.append(value)
    for value in [*expanded_objects, *OPEN_VOCAB_HAND_CLASSES]:
        prompt = str(value).strip()
        key = prompt.casefold()
        if not prompt or key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
    return prompts


def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def _dedupe_open_vocab_detections(detections: list[dict]) -> list[dict]:
    output: list[dict] = []
    for candidate in sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        if any(
            existing.get("group") == candidate.get("group")
            and _box_iou(existing["box"], candidate["box"]) >= 0.72
            for existing in output
        ):
            continue
        output.append(candidate)
    return output


@dataclass
class LocalStatus:
    loaded: bool = False
    loading: bool = False
    kind: str | None = None
    model_path: str | None = None
    device: str = "cpu"
    confidence: float = 0.25
    family: str | None = None
    warmup_ms: float | None = None
    error: str | None = None


@dataclass
class VLMStatus:
    configured: bool = False
    endpoint: str | None = None
    model: str | None = None
    error: str | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local_model: Any = None
        self._local = LocalStatus()
        self._vlm = VLMStatus()
        self._vlm_key: str | None = None
        self._current_classes: tuple[str, ...] = ()
        self._class_embedding_cache: dict[tuple[str, ...], Any] = {}
        self._loader_thread: threading.Thread | None = None

    def status(self) -> dict:
        return {"local": dict(vars(self._local)), "vlm": dict(vars(self._vlm))}

    def configure_local_async(self, config: LocalModelConfig) -> dict:
        """Load and warm the local model without delaying API readiness."""
        with self._lock:
            if self._local.loading:
                return self.status()
            self._local = LocalStatus(
                loading=True,
                kind=config.kind,
                model_path=config.model_path,
                device=config.device,
                confidence=config.confidence,
            )
            self._loader_thread = threading.Thread(
                target=self._configure_local_worker,
                args=(config,),
                name="alice-model-loader",
                daemon=True,
            )
            self._loader_thread.start()
        return self.status()

    def _configure_local_worker(self, config: LocalModelConfig) -> None:
        try:
            self.configure_local(config)
        except RuntimeError:
            # The failure is retained in LocalStatus and exposed by /api/health.
            return

    def configure_local(self, config: LocalModelConfig) -> dict:
        with self._lock:
            self._local = LocalStatus(loading=True, kind=config.kind, model_path=config.model_path, device=config.device, confidence=config.confidence)
            try:
                import torch
                from ultralytics import SAM, YOLO

                device = config.device
                if device == "auto":
                    device = "0" if torch.cuda.is_available() else "cpu"
                if device not in {"cpu", "mps"} and not (device.isdigit() and torch.cuda.is_available()):
                    raise RuntimeError(f"请求的设备 {device} 不可用")

                model_path = config.model_path.strip()
                if not model_path:
                    raise ValueError("模型路径不能为空")
                path = Path(model_path).expanduser()
                if not path.is_absolute():
                    project_path = ROOT / path
                    if project_path.exists():
                        path = project_path
                # Ultralytics model names are allowed and are downloaded by its verified asset loader.
                resolved = str(path.resolve()) if path.exists() else model_path
                model = SAM(resolved) if config.kind == "sam" else YOLO(resolved, task="segment", verbose=False)
                torch_device = f"cuda:{device}" if device.isdigit() else device
                model.to(torch_device)
                family = type(model).__name__
                if family == "YOLOE" and len(model.names) == 80 and all(str(name).isdigit() for name in model.names.values()):
                    model.model.names = dict(enumerate(COCO80))
                    if model.predictor:
                        model.predictor.model.names = model.model.names
                warmup_frame = np.zeros((320, 320, 3), dtype=np.uint8)
                warmup_start = time.perf_counter()
                if config.kind == "sam":
                    model.predict(source=warmup_frame, bboxes=[[64, 64, 256, 256]], device=device, verbose=False)
                else:
                    model.predict(source=warmup_frame, device=device, imgsz=320, conf=config.confidence, verbose=False)
                warmup_ms = round((time.perf_counter() - warmup_start) * 1000, 1)
                self._local_model = model
                self._current_classes = tuple(str(value) for value in model.names.values())
                self._class_embedding_cache = {}
                self._local = LocalStatus(loaded=True, kind=config.kind, model_path=resolved, device=device, confidence=config.confidence, family=family, warmup_ms=warmup_ms)
            except Exception as exc:
                self._local_model = None
                self._current_classes = ()
                self._class_embedding_cache = {}
                self._local.loading = False
                self._local.error = str(exc)
                raise RuntimeError(f"本地模型加载失败: {exc}") from exc
        return self.status()

    def configure_vlm(self, config: VLMModelConfig) -> dict:
        endpoint = config.endpoint.rstrip("/")
        candidate = VLMStatus(False, endpoint, config.model, None)
        try:
            if config.verify:
                self._qwen_request(
                    endpoint=endpoint,
                    api_key=config.api_key,
                    model=config.model,
                    content=[{"type": "text", "text": "Return only this JSON: {\"ok\": true}"}],
                    max_tokens=32,
                )
            candidate.configured = True
            self._vlm_key = config.api_key
            self._vlm = candidate
        except Exception as exc:
            candidate.error = str(exc)
            self._vlm = candidate
            self._vlm_key = None
            raise RuntimeError(f"Qwen-VLM 连接失败: {exc}") from exc
        return self.status()

    @property
    def has_local(self) -> bool:
        return self._local.loaded and self._local_model is not None

    @property
    def has_vlm(self) -> bool:
        return self._vlm.configured and bool(self._vlm_key)

    def infer_local(self, frame: np.ndarray, motion_box: list[int] | None = None) -> dict:
        if not self.has_local:
            raise RuntimeError("尚未加载本地分割模型")
        with self._lock:
            if self._local.kind == "sam":
                return self._infer_sam(frame, motion_box)
            if self._local.family == "YOLOE":
                self._set_yoloe_classes(COCO80)
            return self._infer_yolo(frame)

    def _set_yoloe_classes(self, classes: list[str]) -> None:
        normalized = tuple(dict.fromkeys(str(value).strip() for value in classes if str(value).strip()))
        if not normalized or normalized == self._current_classes:
            return
        embeddings = self._class_embedding_cache.get(normalized)
        if embeddings is None:
            if str(getattr(self._local_model.model, "text_model", "")).startswith("mobileclip2"):
                encoder_path = ROOT / "mobileclip2_b.ts"
                if not encoder_path.is_file() or encoder_path.stat().st_size < 100_000_000:
                    raise RuntimeError("YOLOE 开放词汇编码器 mobileclip2_b.ts 缺失或不完整")
            embeddings = self._local_model.get_text_pe(list(normalized))
            self._class_embedding_cache[normalized] = embeddings
        self._local_model.set_classes(list(normalized), embeddings)
        self._current_classes = normalized

    def infer_open_vocab_proximity(self, frame: np.ndarray, object_classes: list[str], proximity_threshold: float = 0.04) -> dict:
        if not self.has_local or self._local.family != "YOLOE":
            raise RuntimeError("无动作剪切需要已加载的 YOLOE 分割模型")
        objects = [value for value in normalize_yoloe_object_terms(object_classes) if value.casefold() not in _OPEN_VOCAB_HAND_KEYS]
        if not objects:
            raise ValueError("VLM 主要词文件中没有可用物体名词")
        hand_keys = {value.casefold() for value in OPEN_VOCAB_HAND_CLASSES}
        classes = _open_vocab_proximity_classes(objects)
        cpu = str(self._local.device).casefold() == "cpu"
        primary_size = _bounded_env_int(
            "VLA_YOLOE_OPEN_VOCAB_IMGSZ",
            640 if cpu else OPEN_VOCAB_DEFAULT_IMAGE_SIZE,
            320,
            1600,
        )
        fallback_size = _bounded_env_int(
            "VLA_YOLOE_OPEN_VOCAB_FALLBACK_IMGSZ",
            960 if cpu else OPEN_VOCAB_FALLBACK_IMAGE_SIZE,
            primary_size,
            1920,
        )
        configured_confidence = max(0.01, float(self._local.confidence or 0.25))
        primary_confidence = min(
            configured_confidence,
            _bounded_env_float("VLA_YOLOE_OPEN_VOCAB_CONF", OPEN_VOCAB_DEFAULT_CONFIDENCE, 0.01, 0.5),
        )
        fallback_confidence = min(
            primary_confidence,
            _bounded_env_float("VLA_YOLOE_OPEN_VOCAB_FALLBACK_CONF", 0.06, 0.01, 0.25),
        )

        def collect(result) -> list[dict]:
            names = result.names or {}
            boxes = result.boxes
            masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else np.empty((0,))
            collected: list[dict] = []
            if boxes is None:
                return collected
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            for index, (box, confidence, class_id) in enumerate(zip(xyxy, confs, class_ids)):
                label = str(names.get(int(class_id), class_id))
                mask = None
                if index < len(masks):
                    mask = cv2.resize(masks[index], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
                collected.append({
                    "label": label,
                    "confidence": round(float(confidence), 4),
                    "box": [round(float(value), 2) for value in box],
                    "group": "hand" if label.casefold() in hand_keys or "hand" in label.casefold() or "gripper" in label.casefold() else "object",
                    "mask": mask,
                })
            return collected

        with self._lock:
            self._set_yoloe_classes(classes)
            primary_result = self._local_model.predict(
                source=frame,
                device=self._local.device,
                conf=primary_confidence,
                imgsz=primary_size,
                verbose=False,
            )[0]
            detections = collect(primary_result)
            missing_pair = not any(item["group"] == "hand" for item in detections) or not any(item["group"] == "object" for item in detections)
            if missing_pair and fallback_size > primary_size:
                fallback_result = self._local_model.predict(
                    source=frame,
                    device=self._local.device,
                    conf=fallback_confidence,
                    imgsz=fallback_size,
                    verbose=False,
                )[0]
                detections.extend(collect(fallback_result))
            detections = _dedupe_open_vocab_detections(detections)
        hands = [item for item in detections if item["group"] == "hand"]
        targets = [item for item in detections if item["group"] == "object"]
        diagonal = max(1.0, math.hypot(frame.shape[1], frame.shape[0]))
        nearest = None
        for hand in hands:
            for target in targets:
                if hand["mask"] is not None and target["mask"] is not None and hand["mask"].any() and target["mask"].any():
                    distance_field = cv2.distanceTransform((~hand["mask"]).astype(np.uint8), cv2.DIST_L2, 3)
                    distance_px = float(np.min(distance_field[target["mask"]]))
                    distance_source = "segmentation_mask"
                else:
                    hx1, hy1, hx2, hy2 = hand["box"]
                    ox1, oy1, ox2, oy2 = target["box"]
                    dx = max(hx1 - ox2, ox1 - hx2, 0.0)
                    dy = max(hy1 - oy2, oy1 - hy2, 0.0)
                    distance_px = math.hypot(dx, dy)
                    distance_source = "bounding_box"
                normalized_distance = distance_px / diagonal
                candidate = {
                    "hand": hand["label"],
                    "object": target["label"],
                    "distance_px": round(distance_px, 3),
                    "normalized_distance": round(normalized_distance, 6),
                    "distance_source": distance_source,
                    "confidence": round(min(hand["confidence"], target["confidence"]), 4),
                }
                if nearest is None or candidate["normalized_distance"] < nearest["normalized_distance"]:
                    nearest = candidate
        close = bool(nearest and nearest["normalized_distance"] <= proximity_threshold)
        serializable = [{key: value for key, value in item.items() if key != "mask"} for item in detections]
        return {
            "detections": serializable,
            "hand_count": len(hands),
            "object_count": len(targets),
            "nearest": nearest,
            "close": close,
            "proximity_threshold": proximity_threshold,
            "inference": {
                "prompt_classes": classes,
                "confidence": primary_confidence,
                "image_size": primary_size,
                "fallback_used": bool(missing_pair and fallback_size > primary_size),
                "fallback_confidence": fallback_confidence,
                "fallback_image_size": fallback_size,
            },
        }

    def _infer_yolo(self, frame: np.ndarray) -> dict:
        result = self._local_model.predict(
            source=frame,
            device=self._local.device,
            conf=self._local.confidence,
            imgsz=640,
            verbose=False,
        )[0]
        names = result.names or {}
        detections: list[dict] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            for box, conf, class_id in zip(xyxy, confs, classes):
                detections.append({
                    "label": str(names.get(int(class_id), class_id)),
                    "confidence": round(float(conf), 4),
                    "box": [round(float(value), 2) for value in box],
                })
        masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else np.empty((0,))
        mask_coverage = float(np.mean(np.any(masks > 0.5, axis=0))) if len(masks) else 0.0
        interaction = self._interaction_score(detections, frame.shape[1], frame.shape[0])
        return {"detections": detections, "mask_coverage": round(mask_coverage, 4), "interaction": round(interaction, 4)}

    def _infer_sam(self, frame: np.ndarray, motion_box: list[int] | None) -> dict:
        height, width = frame.shape[:2]
        if motion_box is None:
            motion_box = [int(width * 0.25), int(height * 0.2), int(width * 0.75), int(height * 0.85)]
        result = self._local_model.predict(
            source=frame,
            bboxes=[motion_box],
            device=self._local.device,
            verbose=False,
        )[0]
        masks = result.masks.data.detach().cpu().numpy() if result.masks is not None else np.empty((0,))
        coverage = float(np.mean(np.any(masks > 0.5, axis=0))) if len(masks) else 0.0
        detections = [{"label": "sam_region", "confidence": 1.0, "box": motion_box}] if len(masks) else []
        return {"detections": detections, "mask_coverage": round(coverage, 4), "interaction": min(1.0, round(coverage * 8.0, 4))}

    @staticmethod
    def _interaction_score(detections: list[dict], width: int, height: int) -> float:
        if not detections:
            return 0.0
        hands = [item for item in detections if item["label"].lower() in HAND_LABELS or "hand" in item["label"].lower()]
        objects = [item for item in detections if item not in hands]
        if not hands:
            return min(0.45, sum(item["confidence"] for item in detections) / max(1, len(detections)) * 0.35)
        if not objects:
            return 0.25
        diagonal = math.hypot(width, height)
        best = 0.0
        for hand in hands:
            hx1, hy1, hx2, hy2 = hand["box"]
            hc = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
            for obj in objects:
                ox1, oy1, ox2, oy2 = obj["box"]
                oc = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
                distance = math.hypot(hc[0] - oc[0], hc[1] - oc[1]) / max(1.0, diagonal)
                expanded_overlap = not (hx2 + width * 0.04 < ox1 or ox2 + width * 0.04 < hx1 or hy2 + height * 0.04 < oy1 or oy2 + height * 0.04 < hy1)
                score = 0.9 if expanded_overlap else max(0.0, 1.0 - distance * 5.0)
                best = max(best, score * min(hand["confidence"], obj["confidence"]))
        return best

    def judge_frames(self, frames: list[np.ndarray], context: str) -> dict:
        if not self.has_vlm:
            raise RuntimeError("尚未配置 Qwen-VLM")
        content: list[dict] = [{"type": "text", "text": (
            "你是VLA机器人数据质检器。判断连续画面中的手或夹爪是否正在执行对任务有意义的物体操作。"
            "仅输出JSON，字段为 state(valid/invalid/uncertain), confidence(0-1), reason(简短中文), "
            "motion(0-1), contact(0-1)。空闲、等待、手离开工作区、只有相机抖动均为invalid。"
            f"上下文: {context}"
        )}]
        for frame in frames[:6]:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ok:
                continue
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        return self._request_json(self._vlm.endpoint or "", self._vlm_key or "", self._vlm.model or "", content, 180)

    def understand_dataset_schema(self, inventory: dict) -> dict:
        if not self.has_vlm:
            raise RuntimeError("Qwen-VLM is not configured")
        compact_inventory = self._compact_schema_inventory(inventory)
        prompt = (
            "You are a robotics VLA dataset schema analyst. Analyze only the supplied, mechanically inspected inventory. "
            "Do not invent paths or fields. Identify dataset family, episode organization, vision streams, joint/proprio streams, "
            "left/right/bimanual separation, multi-camera streams, pressure/force/torque/tactile sensors, actions and timestamps. "
            "Associate each vision stream with the joint and sensor streams representing the same hand/arm and explain time alignment. "
            "Every source_id must exactly match an id from candidate_streams. Return JSON only with this schema: "
            "{format_family:string,format_confidence:number,summary:string,episode_organization:string,"
            "streams:[{source_id:string,kind:'vision|joint|sensor|action|timestamp',modality:string,"
            "side:'left|right|shared|unknown',role:string,representation:'absolute|delta|velocity|unknown',"
            "dimension_names:[string],gripper_indices:[int],embodiment_id:string|null,confidence:number,evidence:string}],"
            "associations:[{vision_id:string,joint_ids:[string],sensor_ids:[string],side:'left|right|shared|unknown',"
            "time_alignment:string,timestamp_id:string|null,confidence:number,reason:string}],warnings:[string]}.\n"
            "INVENTORY:\n" + json.dumps(compact_inventory, ensure_ascii=False, separators=(",", ":"))
        )
        return self._request_json(
            endpoint=self._vlm.endpoint or "",
            api_key=self._vlm_key or "",
            model=self._vlm.model or "",
            content=[{"type": "text", "text": prompt}],
            max_tokens=6000,
        )

    def annotate_behavior(self, frames: list[tuple[int, float, np.ndarray]], ontology: list[dict], context: str) -> dict:
        if not self.has_vlm:
            raise RuntimeError("Qwen-VLM is not configured")
        compact_ontology = [{
            "label": item["label"],
            "verbs": item.get("verbs", []),
            "objects": item.get("objects", []),
            "descriptions": item.get("descriptions", [])[:4],
        } for item in ontology]
        phase_labels = [
            "idle", "observe", "reach", "grasp", "lift", "transport", "align", "place",
            "release", "withdraw", "manipulate", "inspect", "unknown",
        ]
        phase_guide = {
            "idle": "no task-relevant motion or waiting",
            "observe": "look at the scene before approaching a target",
            "reach": "hand or gripper approaches the target before secure control",
            "grasp": "contact and closure establish control of the target",
            "lift": "the controlled object leaves its support surface",
            "transport": "the held object moves toward its destination",
            "align": "fine position or orientation adjustment near the destination",
            "place": "the object makes supported contact at its destination",
            "release": "the hand or gripper relinquishes control",
            "withdraw": "the hand or gripper moves away after release",
            "manipulate": "task-specific continuous interaction not covered above",
            "inspect": "post-action checking of the result",
            "unknown": "visual evidence is insufficient for a safer phase",
        }
        prompt = (
            "You are annotating manipulation behavior in a robotics video. Use the supplied controlled behavior vocabulary and concise imperative English style. "
            "Choose task_label from ontology labels only, or 'other' when no label fits. "
            "task_label is the high-level task and must remain unchanged across the Episode even when a phase repeats. "
            "Divide the full video into ordered, non-overlapping, variable-duration manipulation phases that cover every frame from the first supplied frame through the last supplied frame. "
            "Use visible transition evidence to choose boundaries; never split mechanically into fixed one-second windows. Repeat phases when the task contains multiple manipulation cycles, and explicitly label waiting or no task-relevant motion as idle. "
            "Start a new segment whenever the manipulated or task-relevant object identity changes during a long operation, even when phase_label remains the same. Never merge adjacent actions that operate on different objects. "
            "Set target_instance to a stable scene-local identifier such as 'block#1' or 'cup#2' for each manipulated object instance; reuse it while operating on the same instance, change it when the object changes, and use an empty string for idle, observe, inspect, or unknown when no single manipulated instance applies. "
            f"Every segment phase_label must be exactly one of {phase_labels}; set label equal to phase_label for compatibility. Use unknown rather than inventing another phase. "
            f"PHASE DEFINITIONS: {json.dumps(phase_guide, ensure_ascii=False, separators=(',', ':'))}. "
            "object_nouns must list every distinct visible physical object noun mentioned in behavior_description, including manipulated objects, containers, tools, work surfaces, and robot/human hands; use concise lowercase English noun phrases. "
            "For object_nouns and primary_targets, emit category-level core object names only. Put color, quantity, size, material, position, action, and relationship modifiers in description or evidence, never inside object names. "
            "primary_targets must contain only objects visibly present in the supplied frames and necessary to perform the behavior. Each target must cite visible_evidence_frames using supplied frame indices. "
            "Return strict JSON only with schema: {task_label:string,direction:'forward|reverse|unknown',behavior_description:string,confidence:number," 
            "segments:[{start_frame:int,end_frame:int,phase_label:'idle|observe|reach|grasp|lift|transport|align|place|release|withdraw|manipulate|inspect|unknown',label:string,target_instance:string,description:string,confidence:number,primary_targets:[string]}],"
            "object_nouns:[string],primary_targets:[{name:string,role:string,confidence:number,visible_evidence_frames:[int],evidence:string}],warnings:[string]}. "
            f"Dataset context: {context}. ONTOLOGY: {json.dumps(compact_ontology, ensure_ascii=False, separators=(',', ':'))}"
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for frame_index, timestamp, frame in frames:
            content.append({"type": "text", "text": f"FRAME {frame_index} TIME {timestamp:.3f}s"})
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
            if not ok:
                continue
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        return self._request_json(
            endpoint=self._vlm.endpoint or "",
            api_key=self._vlm_key or "",
            model=self._vlm.model or "",
            content=content,
            max_tokens=6000,
        )

    @staticmethod
    def _compact_schema_inventory(inventory: dict) -> dict:
        def path_pattern(value: Any) -> str:
            return re.sub(r"(?i)(^|/)(?:ep|episode)[_-]?\d+(?:[_-]\d+)*", r"\1{episode}", str(value or ""))

        representatives: dict[tuple, dict] = {}
        for stream in inventory.get("candidate_streams", []):
            if stream.get("kind") not in {"vision", "joint", "sensor", "action", "timestamp"}:
                continue
            shape = stream.get("shape")
            shape_pattern = ["*", *shape[1:]] if isinstance(shape, list) and shape else shape
            field = str(stream.get("field") or "")
            field_pattern = field
            if str(stream.get("source_path", "")).lower().endswith(('.json', '.jsonl')) and "." in field:
                parts = field.split(".")
                field_pattern = ".".join(parts[:3] if parts[0] == "sensors" else parts[:2])
            signature = (
                path_pattern(stream.get("source_path")), field_pattern,
                str(stream.get("kind") or ""), str(stream.get("modality") or ""),
                str(stream.get("side_hint") or ""),
            )
            if signature not in representatives:
                representative = dict(stream)
                representative["path_pattern"] = signature[0]
                representative["field_pattern"] = field_pattern
                representative["shape_pattern"] = shape_pattern
                representative["equivalent_count"] = 1
                representatives[signature] = representative
            else:
                representatives[signature]["equivalent_count"] += 1

        episodes = inventory.get("episodes", [])
        candidates = list(representatives.values())
        return {
            "root_name": inventory.get("root_name"),
            "file_count": inventory.get("file_count", 0),
            "files_profiled": inventory.get("files_profiled", 0),
            "field_count": inventory.get("field_count", 0),
            "extension_counts": inventory.get("extension_counts", {}),
            "sampling": inventory.get("sampling", {}),
            "episode_count": len(episodes),
            "representative_episodes": episodes[:3],
            "candidate_stream_count": len(inventory.get("candidate_streams", [])),
            "candidate_pattern_count": len(candidates),
            "candidate_streams": candidates,
            "note": "candidate_streams come from folder/extension-stratified samples and are grouped by episode-invariant path/field pattern; source_id always names a mechanically inspected real stream",
        }

    def resolve_episode_membership(self, framework: dict) -> dict:
        if not self.has_vlm:
            raise RuntimeError("Qwen-VLM is not configured")
        prompt = (
            "You are the guarded episode-membership auditor for a robotics VLA dataset. "
            "Group synchronized files using only the supplied file_id values and metadata. "
            "Different non-null episode_token values are hard boundaries. Matching task scope and numeric episode "
            "are strong evidence. Multiple cameras, left/right hand data, joints, actions and tactile sensors may "
            "share an episode only when path/token evidence agrees. Dataset-level metadata must go to "
            "shared_file_ids. Put uncertainty in unassigned_file_ids instead of guessing. Return strict JSON "
            "matching output_schema.\nFRAMEWORK:\n"
            + json.dumps(framework, ensure_ascii=False, separators=(",", ":"))[:70000]
        )
        return self._request_json(
            endpoint=self._vlm.endpoint or "",
            api_key=self._vlm_key or "",
            model=self._vlm.model or "",
            content=[{"type": "text", "text": prompt}],
            max_tokens=5000,
        )

    @staticmethod
    def _request_json(endpoint: str, api_key: str, model: str, content: list[dict], max_tokens: int) -> dict:
        text = ModelRegistry._qwen_request(endpoint, api_key, model, content, max_tokens)
        try:
            return ModelRegistry._extract_json(text)
        except RuntimeError as first_error:
            retry_content = list(content) + [{
                "type": "text",
                "text": "The previous response was not valid JSON. Return the same answer again as compact strict JSON only. Do not use Markdown, comments, trailing commas, or explanatory text. Keep optional warnings and reasons short.",
            }]
            try:
                retry_text = ModelRegistry._qwen_request(endpoint, api_key, model, retry_content, min(12000, max_tokens * 2))
                return ModelRegistry._extract_json(retry_text)
            except RuntimeError as retry_error:
                raise RuntimeError(f"Qwen JSON parse failed after retry: {retry_error}") from first_error

    @staticmethod
    def _qwen_request(endpoint: str, api_key: str, model: str, content: list[dict], max_tokens: int) -> str:
        url = urljoin(endpoint.rstrip("/") + "/", "chat/completions")
        timeout_seconds = max(30.0, float(os.getenv("VLA_QWEN_TIMEOUT", "300")))
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": max_tokens, "enable_thinking": False, "response_format": {"type": "json_object"}},
            timeout=httpx.Timeout(timeout_seconds, connect=min(30.0, timeout_seconds)),
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json_legacy(text: str) -> dict:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end + 1])
            raise RuntimeError(f"VLM 未返回有效 JSON: {text[:200]}")


    @staticmethod
    def _extract_json(text: str) -> dict:
        cleaned = text.strip().replace("```json", "").replace("```JSON", "").replace("```", "").strip()
        decoder = json.JSONDecoder()

        def decode_object(source: str) -> dict | None:
            for start, char in enumerate(source):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(source[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
            return None

        parsed = decode_object(cleaned)
        if parsed is not None:
            return parsed
        repaired = ModelRegistry._repair_json(cleaned)
        parsed = decode_object(repaired)
        if parsed is not None:
            return parsed
        raise RuntimeError(f"VLM returned invalid JSON: {text[:240]}")

    @staticmethod
    def _repair_json(text: str) -> str:
        """Repair common model formatting defects before schema validation."""
        normalized = text.translate(str.maketrans({"“": '"', "”": '"', "„": '"', "’": "'"}))
        start = normalized.find("{")
        if start >= 0:
            normalized = normalized[start:]
        normalized = re.sub(r",(\s*[}\]])", r"\1", normalized)
        normalized = re.sub(r"([}\]])\s*(?=\"[^\"\n]+\"\s*:)", r"\1,", normalized)
        normalized = re.sub(r"(\d|true|false|null)\s+(?=\"[^\"\n]+\"\s*:)", r"\1,", normalized)
        normalized = re.sub(r"([}\]])\s*(?=[{])", r"\1,", normalized)

        stack: list[str] = []
        output: list[str] = []
        in_string = False
        escaped = False
        for char in normalized:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                elif char in "\r\n":
                    output.append("\\n")
                    continue
            elif char == '"':
                in_string = True
            elif char in "[{":
                stack.append("]" if char == "[" else "}")
            elif char in "]}":
                if stack and char == stack[-1]:
                    stack.pop()
                else:
                    continue
            output.append(char)
        if in_string:
            output.append('"')
        output.extend(reversed(stack))
        return "".join(output)


registry = ModelRegistry()
