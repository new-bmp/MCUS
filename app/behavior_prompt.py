from __future__ import annotations

from typing import Any


TRI_LEVEL_PROTOCOL_VERSION = "v3"
TRI_LEVEL_PROTOCOL_SCHEMA = "tri_level_v1"

META_ACTION_TRANSLATIONS = {
    "Move": "移动",
    "TurnWheel": "转动轮子",
    "Turn": "转向",
    "Transport": "运输",
    "Carry": "携带",
    "Raise height": "升高高度",
    "Lower height": "降低高度",
    "Arch waist backward": "腰部后仰",
    "Bend waist forward": "腰部前倾",
    "Turn head": "转头",
    "Grasp": "抓取",
    "Hold": "握持",
    "Place": "放置",
    "Release": "释放",
    "Drop": "掉落",
    "Lift": "举起",
    "Push": "推",
    "Pull": "拉",
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
    "Brush": "刷",
    "Sweep": "扫地",
    "Mop": "拖地",
    "Vacuum": "吸尘",
    "Rinse": "冲洗",
    "Hammer": "锤击",
    "Screw": "拧螺丝",
    "Unscrew": "拧松螺丝",
    "Cut": "切割",
    "Chop": "剁碎",
    "Peel": "削皮",
    "Iron": "熨烫",
    "Stamp": "盖章",
    "Paint": "涂刷",
    "Water": "浇水",
    "Stick": "粘贴",
    "Unzip": "拉开拉链",
    "Tie": "系紧",
    "Untie": "解开",
    "Plug": "插插头",
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
    "Pipette": "吸取液体",
    "Clip": "夹取",
    "HandOver": "递交",
    "TakeOver": "接过",
    "KeepPumping": "持续泵送",
    "Other": "其他",
}

SYSTEM_PROMPT_TEMPLATE = """## 角色定义
你是一位专业的机器人动作分析与时序分割专家。你能够对机器人操作视频进行精确的时序分割，识别动作边界，并使用标准化的动作标签进行标注。

## 任务说明
你将收到一段机器人操作视频（你看到的是从原视频均匀抽样的一组关键帧 / 采样视频）。请仔细观察连续帧之间机器人的运动变化，对视频进行**三级粒度**的时序分割与动作标注。

### 帧索引约定（非常重要）
- **原视频总帧数 = {video_length} frames（时长约 {duration} 秒）**
- 你**看到的是抽样版本**（约 {n_sampled} 帧 / 帧索引 0..{n_sampled_minus_1}），但 `start_frame` / `end_frame` 必须输出在**原视频帧空间**上，范围 `[0, {video_length_minus_1}]`。
- 第一个片段必须从 `0` 开始，最后一个片段必须以 `{video_length_minus_1}` 结束。
- 抽样帧 `i`（0-indexed）大致对应原视频帧 `round(i × ({video_length_minus_1} / {n_sampled_minus_1}))`；输出时按原视频帧标号。

---

### 第一级：粗粒度（Coarse）
**目标**：用一个短语概括机器人在整段视频中执行的任务。
- 不需要帧区间
- 描述任务目标而非具体动作
- 示例："调制饮品"、"整理货架"、"桌面物品分拣"、"厨房清洁"、"零件装配"

---

### 第二级：中间粒度（Medium）
**目标**：将视频切分为若干**语义完整的子任务段**，每段对应一个有明确目的的操作流程。
- 切分依据：当机器人的**操作目标**发生切换时，划分新片段
- 片段首尾严格相接，无空隙、无重叠
- 第一个片段从 frame 0 开始，最后一个片段结束于 {video_length_minus_1}
- 通常 3~8 个片段为宜
- 描述要求：一句话说明操作内容："[哪只臂/双臂] + 做了什么 + 对什么物体 + 从哪里到哪里"
- 需要包含物体名称和空间位置信息

---

### 第三级：细粒度（Fine）
**目标**：将视频切分为**原子级动作段**，每段对应一个 meta_action 标签。
- 切分依据：当机器人执行的**动作类型**（skill）发生变化时，划分新片段
- 片段首尾严格相接，无空隙、无重叠
- 第一个片段从 frame 0 开始，最后一个片段结束于 {video_length_minus_1}
- **时间长度约束：每个 Fine 片段原则上至少覆盖 5 秒视频内容**；不要输出 1~2 秒的碎片段
- 若某个动作真实持续不足 5 秒，应优先与前后语义连续、目标一致的动作合并，skill 选择主要动作
- 只有视频开头/结尾残段，或确实无法合并且对任务理解关键的动作，才允许短于 5 秒
- 准备动作（如伸手靠近）与主动作（如抓取）无缝衔接时需合并
- 每个片段只对应一个主要 skill 标签
- 描述要求：包含动作主体（左臂/右臂/双臂/机器人整体）+ 具体动作内容
- skill 标签必须从下方的 meta_action 列表中选取

## Meta Action 标签字典（必须从中选取 skill）
{meta_actions_formatted}

## 动作类型判定规则

### 身体动作 vs 手臂动作
- 身体动作：机器人整体发生位移或姿态变化，视频表现为背景整体移动/旋转
- 仅以下标签可用于身体动作：Move, TurnWheel, Turn, Raise height, Lower height, Arch waist backward, Bend waist forward, Turn head
- 手臂动作：机器人左/右/双臂运动，背景静止，仅肢体和末端执行器移动
- 除上述身体动作标签外的所有标签均用于手臂动作
- 手臂动作描述中必须写明是左臂、右臂还是双臂

### 动作合并规则
- 准备动作（伸手靠近目标）+ 主动作（抓取）→ 合并为一个片段，skill 取主动作
- 同一动作的持续执行（如持续倾倒 2 秒）→ 不拆分
- 同一手臂连续对同一物体的操作序列中，动作类型未变 → 不拆分
- 为保证时间边界稳定，Fine 片段不要过度切分；相邻短动作如果服务于同一操作目标，应合并成不少于 5 秒的片段
- 合并后的片段仍只填写一个主要 skill，选择最能代表该片段目的的动作标签

### 常见易混淆场景
- 手臂移动靠近物体 → skill 应为 Grasp（准备+抓取合并），不是 Move
- 手臂携带物体移动到目标位置 → skill 应为 Carry 或 Transport（取决于是否已到达目标）
- 机器人整体前进/后退/转向 → 才用 Move / TurnWheel / Turn
- 夹持器夹取小物体（如叶子、纸片）→ skill 应为 Clip，不是 Grasp
- 将物体从手中递给另一只手 → skill 应为 HandOver

## 正确示例
Frames 0-5: 机器人向前移动靠近货架（背景发生移动）-> Skill: Move
Frames 6-10: 机器人右臂抓取桌面上的水瓶 -> Skill: Grasp
Frames 11-30: 机器人双臂协同将水瓶的瓶盖拧紧 -> Skill: Screw
Frames 31-35: 机器人左臂将水瓶传递至右臂 -> Skill: HandOver
Frames 36-40: 机器人右臂将水瓶放到桌面上 -> Skill: Place
Frames 41-45: 机器人向右转转向桌子（背景发生旋转）-> Skill: TurnWheel

## 错误示例
Frames 0-5: 机器人右臂将勺子移向货架 -> Skill: Move
错误原因：手臂移动不是身体动作，Move 只用于机器人整体位移
Frames 6-10: 机器人将左臂移向绿色瓶子 -> Skill: Move
错误原因：手臂靠近物体是 Grasp 的准备阶段，应合并为 Grasp

## 输出格式
请严格按照以下 JSON 格式输出，用 ```json ... ``` 包裹，不要有额外文字：
```json
{
  "coarse": {
    "summary": "机器人执行的整体任务描述"
  },
  "medium": [
    {
      "start_frame": 0,
      "end_frame": 10,
      "description": "子任务描述（含物体和位置信息）"
    }
  ],
  "fine": [
    {
      "start_frame": 0,
      "end_frame": 3,
      "description": "[左臂/右臂/双臂/机器人] + 动作内容描述",
      "skill": "从meta_action字典中选取的标签（英文）"
    }
  ]
}
```"""

USER_PROMPT_TEMPLATE = """请基于视频内容进行机器人动作三级粒度时序分割。重点保证 Fine 片段时间边界稳定，每段原则上至少 5 秒；短动作优先合并到相邻主动作。输出必须是 ```json ... ``` 包裹的有效 JSON。"""


def canonical_meta_action(value: Any) -> str:
    text = str(value or "").strip()
    lookup = {key.casefold(): key for key in META_ACTION_TRANSLATIONS}
    return lookup.get(text.casefold(), "Other")


def build_tri_level_prompts(
    *,
    video_length: int,
    duration: float,
    sampled_frames: list[int],
    context: str = "",
) -> tuple[str, str]:
    frame_count = max(1, int(video_length))
    sample_count = max(1, len(sampled_frames))
    replacements = {
        "{video_length}": str(frame_count),
        "{duration}": f"{max(0.0, float(duration)):.3f}",
        "{n_sampled}": str(sample_count),
        "{n_sampled_minus_1}": str(max(0, sample_count - 1)),
        "{video_length_minus_1}": str(max(0, frame_count - 1)),
        "{meta_actions_formatted}": "\n".join(f"- {key}: {value}" for key, value in META_ACTION_TRANSLATIONS.items()),
    }
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    for placeholder, value in replacements.items():
        system_prompt = system_prompt.replace(placeholder, value)
    exact_mapping = ", ".join(f"{index}->{frame}" for index, frame in enumerate(sampled_frames))
    user_prompt = (
        USER_PROMPT_TEMPLATE
        + f"\n抽样帧到原视频帧的精确映射为：{exact_mapping}。若与均匀抽样近似值冲突，以该精确映射为准。"
        + (f"\n数据集上下文：{context}" if context else "")
    )
    return system_prompt, user_prompt
