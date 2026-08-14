from __future__ import annotations

import json
from typing import Any


TRI_LEVEL_PROTOCOL_VERSION = "v4"
TRI_LEVEL_PROTOCOL_SCHEMA = "tri_level_v1"

META_ACTION_TRANSLATIONS = {
    "Idle": "空闲",
    "Observe": "观察",
    "Reach": "接近目标",
    "Withdraw": "撤回手臂",
    "Align": "对齐",
    "Inspect": "检查",
    "Move": "整体移动",
    "TurnWheel": "转动轮子",
    "Turn": "转向",
    "Transport": "运输",
    "Carry": "携带",
    "Raise height": "升高",
    "Lower height": "降低",
    "Arch waist backward": "腰部后仰",
    "Bend waist forward": "腰部前倾",
    "Turn head": "转头",
    "Grasp": "抓取",
    "Hold": "保持抓取",
    "Place": "放置",
    "Release": "释放",
    "Drop": "掉落",
    "Lift": "举起",
    "Push": "推动",
    "Pull": "拉动",
    "Press": "按压",
    "Tap": "轻敲",
    "Touch": "触摸",
    "Point": "指向",
    "Wave": "挥手",
    "Clap": "拍手",
    "Open": "打开",
    "Close": "关闭",
    "Insert": "插入",
    "Remove": "移除",
    "Takeout": "取出",
    "Fold": "折叠",
    "Unfold": "展开",
    "Roll": "滚动",
    "Flip": "翻转",
    "Twist": "扭转",
    "Rotate": "旋转",
    "Stretch": "拉伸",
    "Squeeze": "挤压",
    "Pinch": "捏取",
    "Wipe": "擦拭",
    "Brush": "刷动",
    "Sweep": "扫动",
    "Mop": "拖动清洁",
    "Vacuum": "吸尘",
    "Rinse": "冲洗",
    "Hammer": "锤击",
    "Screw": "拧紧螺丝",
    "Unscrew": "拧松螺丝",
    "Cut": "切割",
    "Chop": "剁切",
    "Peel": "削皮",
    "Iron": "熨烫",
    "Stamp": "盖章",
    "Paint": "涂刷",
    "Water": "浇水",
    "Stick": "粘贴",
    "Unzip": "拉开拉链",
    "Tie": "系紧",
    "Untie": "解开",
    "Plug": "插接",
    "Toggle": "拨动",
    "Pour": "倾倒",
    "Scoop": "舀取",
    "Stir": "搅拌",
    "Knead": "揉捏",
    "Dip": "浸入",
    "Shake": "摇动",
    "Stack": "堆叠",
    "Unstack": "拆叠",
    "Straighten": "整理",
    "Hang": "悬挂",
    "Throw": "投掷",
    "Catch": "接住",
    "PullOut": "拔出",
    "Scan": "扫描",
    "Suction": "吸附",
    "Pipette": "移液",
    "Clip": "夹取",
    "HandOver": "递交",
    "TakeOver": "接过",
    "KeepPumping": "持续泵送",
    "Other": "其他",
}


def canonical_meta_action(value: Any) -> str:
    text = str(value or "").strip()
    lookup = {key.casefold(): key for key in META_ACTION_TRANSLATIONS}
    return lookup.get(text.casefold(), "Other")


def _ontology_text(ontology: list[dict] | None) -> str:
    compact = []
    for item in ontology or []:
        if not isinstance(item, dict):
            continue
        compact.append({
            "label": str(item.get("label") or "")[:120],
            "task": str(item.get("task") or "")[:240],
            "verbs": [str(value)[:80] for value in (item.get("verbs") or [])[:12]],
            "objects": [str(value)[:120] for value in (item.get("objects") or [])[:12]],
            "descriptions": [str(value)[:300] for value in (item.get("descriptions") or [])[:4]],
        })
    return json.dumps(compact[:80], ensure_ascii=False, separators=(",", ":"))


def build_tri_level_prompts(
    *,
    video_length: int,
    duration: float,
    sampled_frames: list[int],
    context: str = "",
    ontology: list[dict] | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
    window_index: int = 0,
    window_count: int = 1,
) -> tuple[str, str]:
    frame_count = max(1, int(video_length))
    start = max(0, min(frame_count - 1, int(window_start if window_start is not None else 0)))
    end = max(start, min(frame_count - 1, int(window_end if window_end is not None else frame_count - 1)))
    meta_actions = "\n".join(f"- {name}: {translation}" for name, translation in META_ACTION_TRANSLATIONS.items())
    system_prompt = f"""你是机器人操作数据的时序标注专家。

当前请求只分析一个局部时间窗口。输入是按时间顺序排列的独立图片，不是均匀覆盖整个 Episode 的缩略图。每张图前都给出准确的绝对帧号和时间戳。

Episode 总帧数 = {frame_count}，总时长约 {max(0.0, float(duration)):.3f} 秒。
当前窗口 = [{start}, {end}]，窗口编号 {window_index + 1}/{max(1, int(window_count))}。

要求：
1. 只描述当前窗口中图片能够支持的动作，不得推测窗口外内容。
2. start_frame、end_frame、evidence_frames 必须使用 Episode 的绝对帧坐标，并限制在 [{start}, {end}]。
3. Fine 表示原子动作。短暂的 Reach、Touch、Release、Withdraw 必须保留，不得为了凑时长强行合并。
4. 每个 Fine 动作必须输出 confidence、object_nouns、primary_targets、target_instance 和 evidence_frames。
5. evidence_frames 只能选择本次实际提供的采样帧。
6. 若证据不足，使用 Other 或降低 confidence；不要伪造精确边界。
7. 左右手、双手和目标物体发生变化时，应拆分动作。

可用 meta_action：
{meta_actions}

返回严格 JSON，不要 Markdown：
{{
  "window_summary": "当前窗口的简短行为概述",
  "coarse": {{"summary": "对当前窗口所属总体任务的判断"}},
  "medium": [
    {{"start_frame": {start}, "end_frame": {end}, "description": "当前窗口中的中层子任务"}}
  ],
  "fine": [
    {{
      "start_frame": {start},
      "end_frame": {end},
      "description": "动作主体、动作、目标物体和位置",
      "skill": "从 meta_action 中选择",
      "confidence": 0.0,
      "object_nouns": ["可检测的物体名词"],
      "primary_targets": ["主要操作目标"],
      "target_instance": "用于区分同类目标的实例名",
      "evidence_frames": [{start}]
    }}
  ],
  "object_nouns": ["窗口内出现的主要操作物体"],
  "primary_targets": [
    {{"name": "目标名称", "role": "behavior_target", "confidence": 0.0, "visible_evidence_frames": [{start}], "evidence": "可见证据"}}
  ],
  "warnings": []
}}"""
    exact_mapping = ", ".join(f"{index}->{frame}" for index, frame in enumerate(sampled_frames))
    user_prompt = (
        f"请分析当前窗口。采样图序号到 Episode 绝对帧号的精确映射为：{exact_mapping}。"
        f"\n行为 ontology（用于约束任务名称和物体词，不代表画面事实）：{_ontology_text(ontology)}"
        + (f"\n数据集上下文：{context}" if context else "")
    )
    return system_prompt, user_prompt


def build_window_summary_prompt(
    *,
    video_length: int,
    duration: float,
    allowed_ranges: list[tuple[int, int]],
    window_results: list[dict],
    ontology: list[dict] | None = None,
    context: str = "",
) -> str:
    compact_windows = []
    for item in window_results:
        compact_windows.append({
            "window_id": item.get("window_id"),
            "start_frame": item.get("start_frame"),
            "end_frame": item.get("end_frame"),
            "summary": item.get("summary"),
            "fine": [
                {
                    "start_frame": segment.get("start_frame"),
                    "end_frame": segment.get("end_frame"),
                    "skill": segment.get("skill"),
                    "description": segment.get("description"),
                    "primary_targets": segment.get("primary_targets"),
                    "confidence": segment.get("confidence"),
                }
                for segment in (item.get("segments") or [])
            ],
        })
    evidence = json.dumps(compact_windows, ensure_ascii=False, separators=(",", ":"))[:90000]
    return f"""你是机器人操作数据的全局任务归纳专家。下面是多个重叠时间窗口已经完成的局部视觉标注。

Episode 总帧数={max(1, int(video_length))}，时长约={max(0.0, float(duration)):.3f}秒。
允许分析的有效帧区间={allowed_ranges}。区间外是坏帧或质量预检查排除区域，不得根据相邻窗口补写动作。

请完成两件事：
1. 输出一个 Coarse 总任务摘要。
2. 输出多个 Medium 中层子任务及绝对帧边界。Medium 应描述目标切换或完整操作阶段，不要把每个 Fine 都复制成 Medium。

优先使用 ontology 中存在的任务概念，但只能在局部视觉证据支持时使用。
ontology={_ontology_text(ontology)}
上下文={context}
局部窗口结果={evidence}

返回严格 JSON，不要 Markdown：
{{
  "coarse": {{"summary": "总体任务"}},
  "medium": [{{"start_frame": 0, "end_frame": {max(0, int(video_length) - 1)}, "description": "中层子任务"}}],
  "confidence": 0.0,
  "warnings": []
}}"""
