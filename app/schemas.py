from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PathOpenRequest(BaseModel):
    path: str
    name: str | None = None
    analyze_schema: bool = True


class ExportFolderRequest(BaseModel):
    path: str
    include_media: bool = False


class ExcludeFilesRequest(BaseModel):
    file_ids: list[str] = Field(default_factory=list, min_length=1)
    reason: str = Field(default="manual_exclusion", max_length=500)
    scope_type: Literal["file", "episode"] = "file"
    scope_label: str | None = Field(default=None, max_length=500)


class LocalModelConfig(BaseModel):
    slot: Literal["local"] = "local"
    kind: Literal["yolo", "sam"]
    model_path: str
    device: str = "auto"
    confidence: float = Field(default=0.25, ge=0.01, le=1.0)


class VLMModelConfig(BaseModel):
    slot: Literal["vlm"] = "vlm"
    kind: Literal["qwen"] = "qwen"
    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str
    model: str = "qwen2.5-vl-72b-instruct"
    verify: bool = True


class AnalysisRequest(BaseModel):
    sample_fps: float = Field(default=3.0, ge=0.25, le=15.0)
    motion_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    contact_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    idle_duration: float = Field(default=0.8, ge=0.1, le=30.0)
    use_vlm: bool = False
    vlm_window_seconds: float = Field(default=2.0, ge=0.5, le=10.0)


class BehaviorAnnotationRequest(BaseModel):
    sample_count: int = Field(default=18, ge=6, le=24)
    force: bool = False


class CurationJobRequest(BaseModel):
    episode_ids: list[str] = Field(default_factory=list, min_length=1)
    media_file_ids: dict[str, str] = Field(default_factory=dict)
    sudden_change_sigma: float = Field(default=6.0, ge=3.0, le=12.0)
    repair_s1_spikes: bool = True
    s1_max_repair_frames: int = Field(default=5, ge=1, le=15)
    directional_agreement_threshold: float = Field(default=0.65, ge=0.5, le=0.85)
    max_lag_seconds: float = Field(default=0.5, ge=0.1, le=3.0)
    outlier_alpha: float = Field(default=0.1, ge=0.0, le=2.0)
    video_sample_fps: float = Field(default=2.0, ge=0.5, le=8.0)
    black_level_threshold: float = Field(default=8.0, ge=0.0, le=40.0)
    blur_laplacian_threshold: float = Field(default=35.0, ge=5.0, le=300.0)
    static_difference_threshold: float = Field(default=1.5, ge=0.1, le=12.0)
    static_duration_seconds: float = Field(default=2.0, ge=0.5, le=20.0)
    quality_gap_merge_seconds: float = Field(default=0.3, ge=0.0, le=2.0)
    vlm_sample_count: int = Field(default=18, ge=6, le=24)
    force_vlm: bool = False
    full_pipeline: bool = False
    full_output_format: Literal["lerobot", "hdf5_mp4"] = "lerobot"
    full_action_profile_id: str | None = Field(default=None, min_length=1, max_length=80)
    full_action_source_hand: Literal["left", "right"] = "right"
    full_action_coordinate_frame: Literal["camera", "world"] = "camera"
    full_action_horizon_frames: int = Field(default=3, ge=1, le=30)


class ActionMappingRequest(BaseModel):
    episode_ids: list[str] = Field(default_factory=list, min_length=1)
    profile_id: str = Field(default="generic_bimanual_pose", min_length=1, max_length=80)
    source_hand: Literal["left", "right"] = "right"
    coordinate_frame: Literal["camera", "world"] = "camera"
    horizon_frames: int = Field(default=3, ge=1, le=30)
    force: bool = False


class BatchAnalysisRequest(BaseModel):
    operation: Literal["video_smoothing", "vlm_behavior", "pose_recovery", "no_action_trim"]
    episode_ids: list[str] = Field(default_factory=list, min_length=1)
    media_file_ids: dict[str, str] = Field(default_factory=dict)
    sample_count: int = Field(default=18, ge=6, le=24)
    sample_fps: float = Field(default=4.0, ge=0.25, le=15.0)
    proximity_threshold: float = Field(default=0.04, ge=0.005, le=0.25)
    max_gap_seconds: float = Field(default=0.5, ge=0.0, le=3.0)
    min_valid_seconds: float = Field(default=0.3, ge=0.0, le=3.0)
    force: bool = False


class SegmentUpdate(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    state: Literal["valid", "invalid", "uncertain"]
    reason: str = "manual_review"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    media_file_id: str | None = None


class BehaviorPhaseRemovalRequest(BaseModel):
    phase_label: str = Field(min_length=1, max_length=160)
    media_file_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)


class ApplyChangesRequest(BaseModel):
    change_ids: list[str] = Field(default_factory=list, min_length=1)
    confirmation: Literal["APPLY"]
