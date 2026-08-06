from __future__ import annotations

import os
import json
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from .analyzer import jobs
from .action_mapping import action_mapping_jobs, action_mapping_profiles, load_episode_action_mapping
from .annotation_edits import apply_behavior_phase_exclusion, apply_segment_override
from .batch_jobs import batch_analysis_jobs
from .behavior_annotator import behavior_analysis_context, behavior_annotation_status, behavior_jobs, load_behavior_annotation
from .curation_pipeline import curation_jobs, curation_preflight, load_curation_report
from .dataset_format import inspect_dataset_format
from .episode_resolver import build_sampled_episode_framework, validate_qwen_episode_plan
from .file_preview import preview_file, preview_file_frame
from .folder_dialog import choose_folder
from .full_run import full_run_review_media, load_full_run_episode_bundle
from .joint_overlay import draw_joint_overlay, joint_overlay_geometry, joint_overlay_status
from .models import registry
from .no_action_trim import load_no_action_trim
from .pose_recovery import pose_recovery_status, recover_episode_pose
from .projection_correction import preferred_projection_review_media, projection_correction_status
from .preview_proxy import preview_proxy_manager
from .qwen_trim import QwenTrimRequest, load_qwen_action_trim, qwen_trim_jobs
from .schema_profiler import validate_understanding
from .sensor_alignment import load_sensor_alignment, scan_episode_sensor_alignment, sensor_alignment_jobs
from .schemas import ActionMappingRequest, AnalysisRequest, ApplyChangesRequest, BatchAnalysisRequest, BehaviorAnnotationRequest, BehaviorPhaseRemovalRequest, CurationJobRequest, ExcludeFilesRequest, ExportFolderRequest, HandPoseModelConfig, LocalModelConfig, PathOpenRequest, SegmentUpdate, VLMModelConfig
from .storage import (
    ALICE_ANNOTATION_SCHEMA,
    apply_changes,
    ROOT,
    dataset_cache_dir,
    discover_dataset_roots,
    exclude_dataset_files,
    episode_media,
    export_dataset,
    export_zip,
    get_dataset_file,
    get_dataset_file_path,
    get_episode,
    get_manifest,
    list_manifests,
    load_annotations,
    load_invalid_frame_index,
    list_changes,
    is_frame_invalid,
    manifest_registry_path,
    read_frame,
    save_annotations,
    save_manifest,
    scan_dataset,
)


def _default_model_path() -> Path | None:
    configured = os.getenv("VLA_LOCAL_MODEL")
    candidates = [Path(configured)] if configured else []
    candidates.extend([
        ROOT / "yoloe-26x-seg.pt",
        ROOT / "models" / "yoloe-26x-seg.pt",
        Path.home() / "Desktop" / "alice blue" / "yoloe-26x-seg.pt",
        Path.home() / "Desktop" / "yolov26_l" / "yoloe-26x-seg.pt",
    ])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        # A Git LFS pointer is a text stub, not a loadable model.  Skip it and
        # continue to a real local weight file when available.
        if resolved.is_file() and resolved.stat().st_size >= 1024 * 1024:
            return resolved
    return None


def _default_hand_pose_model_path() -> Path | None:
    configured = os.getenv("ALICE_HAND_POSE_MODEL")
    candidates = [Path(configured)] if configured else []
    candidates.extend([
        ROOT / "Alicepose-21k-v1.pt",
        ROOT / "models" / "Alicepose-21k-v1.pt",
        Path.home() / "Desktop" / "alice blue" / "Alicepose-21k-v1.pt",
        Path.home() / "Desktop" / "Alicepose-21k-v1.pt",
    ])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return None


def _vlm_config_path() -> Path:
    return ROOT / ".vla_lens" / "vlm-config.json"


def _save_vlm_config(config: VLMModelConfig) -> None:
    target = _vlm_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "slot": "vlm", "kind": "qwen", "endpoint": config.endpoint,
        "api_key": config.api_key, "model": config.model, "verify": False,
    }, ensure_ascii=False), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(target)


def _load_vlm_config() -> None:
    target = _vlm_config_path()
    if not target.is_file():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["verify"] = False
    registry.configure_vlm(VLMModelConfig.model_validate(payload))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        if os.getenv("ALICE_OPENCV_THREADS"):
            cv2.setNumThreads(max(1, int(os.environ["ALICE_OPENCV_THREADS"])))
    except ValueError:
        pass
    model_path = _default_model_path()
    if model_path and os.getenv("VLA_SKIP_MODEL_AUTOLOAD", "").strip().casefold() not in {"1", "true", "yes"}:
        registry.configure_local_async(
            LocalModelConfig(kind="yolo", model_path=str(model_path), device="auto", confidence=0.25)
        )
    if os.getenv("ALICE_SKIP_HAND_POSE_AUTOLOAD", "").strip().casefold() not in {"1", "true", "yes"}:
        hand_backend = os.getenv("ALICE_HAND_POSE_BACKEND", "mediapipe").strip().casefold()
        hand_pose_path = _default_hand_pose_model_path()
        if hand_backend in {"alicepose", "pose", "yolo"} and hand_pose_path:
            registry.configure_hand_pose_async(
                HandPoseModelConfig(kind="alicepose", model_path=str(hand_pose_path), device="auto", confidence=0.1)
            )
        else:
            registry.configure_hand_pose_async(
                HandPoseModelConfig(kind="mediapipe", model_path="", device="cpu", confidence=0.35)
            )
    try:
        _load_vlm_config()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        pass
    try:
        yield
    finally:
        registry.close()


app = FastAPI(title="alice blue API", version="1.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)


def _understand_manifest(manifest: dict, require_vlm: bool = False) -> dict:
    profile = manifest.get("schema_profile") or {}
    if not registry.has_vlm:
        if require_vlm:
            raise RuntimeError("请先配置 Qwen-VLM，再理解数据集结构")
        return manifest
    try:
        raw = registry.understand_dataset_schema(profile.get("inventory") or {})
        understanding, warnings = validate_understanding(profile.get("inventory") or {}, raw)
        format_map = manifest.get("format_map") or {}
        canonical_family = str(format_map.get("format_family") or "").strip()
        if canonical_family and canonical_family != "unknown":
            proposed_family = str(understanding.get("format_family") or "unknown")
            if proposed_family not in {"", "unknown", canonical_family}:
                warnings.append(
                    f"Qwen format family {proposed_family} was ignored; local preflight confirmed {canonical_family}."
                )
            understanding["format_family"] = canonical_family
            understanding["format_confidence"] = max(
                float(understanding.get("format_confidence") or 0.0),
                float(format_map.get("format_confidence") or 0.0),
            )
        if isinstance(format_map.get("capabilities"), dict):
            understanding["capabilities"] = dict(format_map["capabilities"])
        local_resolution = manifest.get("episode_resolution") or {}
        if local_resolution.get("requires_api"):
            episode_framework = build_sampled_episode_framework(
                manifest.get("files", []),
                focus_ids=set(local_resolution.get("unassigned_file_ids") or []),
            )
            episode_raw = registry.resolve_episode_membership(episode_framework)
            manifest["episode_resolution"] = validate_qwen_episode_plan(
                episode_raw,
                episode_framework,
                manifest.get("files", []),
                manifest.get("episodes", []),
                registry.status().get("vlm", {}).get("model"),
            )
            manifest["episode_resolution"]["sampling"] = episode_framework.get("sampling")
        profile.update({
            "status": "completed",
            "understanding": understanding,
            "warnings": warnings,
            "provider": registry.status()["vlm"],
            "error": None,
        })
    except Exception as exc:
        profile.update({"status": "error", "understanding": None, "error": str(exc)})
        resolution = manifest.get("episode_resolution") or {}
        resolution["status"] = "qwen_error"
        resolution["requires_api"] = True
        resolution["warnings"] = list(resolution.get("warnings", [])) + [f"Qwen episode audit failed: {exc}"]
        manifest["episode_resolution"] = resolution
        if require_vlm:
            manifest["schema_profile"] = profile
            save_manifest(manifest)
            raise RuntimeError(f"Qwen 数据结构理解失败: {exc}") from exc
    manifest["schema_profile"] = profile
    save_manifest(manifest)
    return manifest


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "vla-lens",
        "version": app.version,
        "pid": os.getpid(),
        "instance_id": os.getenv("VLA_INSTANCE_ID"),
        "models": registry.status(),
        "runtime": {"full_pipeline": curation_jobs.runtime_config()},
    }


@app.get("/api/datasets")
def datasets():
    return {"items": list_manifests()}


def _dataset_format_preflight(path: str) -> dict:
    report = inspect_dataset_format(path)
    issues = list(report.get("issues") or [])
    return {
        **report,
        "warnings": [
            item for item in issues
            if str(item.get("severity") or "").casefold() in {"warning", "error"}
        ],
        "episode_summary": {
            "layout": report.get("episode_layout"),
            "count_hint": report.get("episode_count_hint"),
            "samples": list(report.get("episode_samples") or []),
        },
    }


@app.post("/api/datasets/preflight")
def dataset_format_preflight(request: PathOpenRequest):
    """Inspect a source directory without scanning it into the registry."""
    try:
        return _dataset_format_preflight(request.path)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/system/open-dataset-folder")
def open_dataset_folder(preflight_only: bool = True):
    try:
        selected = choose_folder()
        if selected is None:
            return {"cancelled": True, "dataset": None}
        discovery = discover_dataset_roots(selected)
        if discovery["mode"] == "collection":
            return {"cancelled": False, "dataset": None, **discovery}
        # Folder selection is deliberately discovery/preflight-only.  Import
        # must happen through ``open-path`` with the returned confirmation
        # token so blocked or changed formats cannot bypass user review.
        return {
            "cancelled": False,
            "dataset": None,
            "preflight": _dataset_format_preflight(selected),
            **discovery,
        }
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/datasets/{dataset_id}")
def dataset_detail(dataset_id: str):
    try:
        return get_manifest(dataset_id)
    except KeyError:
        raise HTTPException(404, "数据集不存在")


@app.post("/api/datasets/{dataset_id}/rescan")
def rescan_dataset(dataset_id: str):
    try:
        manifest = get_manifest(dataset_id)
        report = _dataset_format_preflight(manifest["root_path"])
        if report.get("root_mode") == "collection":
            raise HTTPException(409, "原数据集目录现在包含多个独立数据集，请重新选择具体子数据集")
        if report.get("status") == "blocked" or not (report.get("capabilities") or {}).get("can_import"):
            raise HTTPException(422, "重新扫描前的格式预检未通过")
        return _understand_manifest(
            scan_dataset(
                manifest["root_path"],
                manifest["name"],
                dataset_id=dataset_id,
                camera_profile_id=(manifest.get("camera_calibration") or {}).get("selected_profile_id"),
            )
        )
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(404, "数据集不存在")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/datasets/{dataset_id}/files/{file_id}")
def dataset_file_detail(dataset_id: str, file_id: str):
    try:
        _, detail = get_dataset_file(dataset_id, file_id)
        return detail
    except KeyError:
        raise HTTPException(404, "文件不在当前数据集索引中")


@app.post("/api/datasets/{dataset_id}/exclusions")
def exclude_files(dataset_id: str, request: ExcludeFilesRequest):
    try:
        manifest = exclude_dataset_files(
            dataset_id,
            request.file_ids,
            request.reason,
            request.scope_type,
            request.scope_label,
        )
        return {"dataset": manifest, "excluded_file_ids": request.file_ids}
    except KeyError as exc:
        raise HTTPException(404, f"文件不在当前数据集索引中: {exc.args[0]}")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/files/{file_id}/preview")
def dataset_file_preview(dataset_id: str, file_id: str, field: str | None = None):
    try:
        _, detail, path = get_dataset_file_path(dataset_id, file_id)
        return preview_file(path, detail["relative_path"], field)
    except KeyError:
        raise HTTPException(404, "文件不在当前数据集索引中")


@app.get("/api/datasets/{dataset_id}/files/{file_id}/frame")
def dataset_file_frame(
    dataset_id: str,
    file_id: str,
    index: int = Query(0, ge=0),
    field: str | None = None,
):
    try:
        _, detail, path = get_dataset_file_path(dataset_id, file_id)
        return preview_file_frame(path, detail["relative_path"], index, field)
    except KeyError:
        raise HTTPException(404, "文件不在当前数据集索引中")


@app.post("/api/datasets/open-path")
def open_path(request: PathOpenRequest, confirmation_token: str | None = None):
    try:
        if not confirmation_token:
            raise HTTPException(428, "请先完成格式预检并确认，再导入数据集")
        report = _dataset_format_preflight(request.path)
        if report.get("confirmation_token") != confirmation_token:
            raise HTTPException(409, "文件夹内容已发生变化，请重新确认数据格式后再导入")
        if report.get("root_mode") == "collection":
            raise HTTPException(409, "该目录包含多个独立数据集，请逐个选择子数据集导入")
        if report.get("status") == "blocked" or not (report.get("capabilities") or {}).get("can_import"):
            raise HTTPException(422, "格式预检未通过，当前文件夹不能安全导入")
        manifest = scan_dataset(
            request.path,
            request.name,
            camera_profile_id=request.camera_profile_id,
        )
        return _understand_manifest(manifest) if request.analyze_schema else manifest
    except HTTPException:
        raise
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/datasets/{dataset_id}/schema")
def dataset_schema(dataset_id: str):
    try:
        return get_manifest(dataset_id).get("schema_profile") or {}
    except KeyError:
        raise HTTPException(404, "数据集不存在")


@app.get("/api/datasets/{dataset_id}/episode-resolution")
def dataset_episode_resolution(dataset_id: str):
    try:
        return get_manifest(dataset_id).get("episode_resolution") or {}
    except KeyError:
        raise HTTPException(404, "数据集不存在")


@app.post("/api/datasets/{dataset_id}/analyze-schema")
def analyze_dataset_schema(dataset_id: str):
    try:
        manifest = _understand_manifest(get_manifest(dataset_id), require_vlm=True)
        result = dict(manifest.get("schema_profile") or {})
        result["episode_resolution"] = manifest.get("episode_resolution") or {}
        return result
    except KeyError:
        raise HTTPException(404, "数据集不存在")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/frame")
def frame(
    dataset_id: str,
    episode_id: str,
    index: int = Query(0, ge=0),
    overlay: bool = False,
    joint_overlay: bool = False,
    joint_indices: bool = False,
    joint_frame_offset: int = Query(0, ge=-30, le=30),
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
    media_file_id: str | None = None,
    full_run_id: str | None = None,
):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        selected_media = episode_media(episode, media_file_id)
        if full_run_id:
            selected_media, _ = full_run_review_media(manifest, episode, selected_media, full_run_id)
        elif joint_mode == "corrected":
            selected_media, _ = preferred_projection_review_media(manifest, episode, selected_media)
    except KeyError:
        raise HTTPException(404, "Episode 不存在")
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc))
    primary_media_id = episode.get("primary_media_file_id")
    if overlay and (not media_file_id or media_file_id == primary_media_id):
        cache_dir = dataset_cache_dir(dataset_id, episode_id)
        candidates = list(cache_dir.glob("*.jpg")) if cache_dir.exists() else []
        if candidates:
            target = min(candidates, key=lambda path: abs(int(path.stem) - index))
            image = cv2.imread(str(target))
        else:
            image = read_frame(selected_media, index)
    else:
        image = read_frame(selected_media, index)
    if image is None:
        raise HTTPException(422, "无法读取指定帧")
    if joint_overlay:
        try:
            joint_count = max(1, int(selected_media.get("frame_count") or episode.get("frame_count") or 1))
            joint_index = max(0, min(joint_count - 1, index + joint_frame_offset))
            image, _ = draw_joint_overlay(image, manifest, episode, joint_index, selected_media, show_indices=joint_indices, mode=joint_mode)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "帧编码失败")
    return Response(encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _preview_proxy_payload(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None,
    start: bool = False,
    force_proxy: bool = False,
    joint_mode: str = "auto",
    full_run_id: str | None = None,
) -> dict:
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        selected_media = episode_media(episode, media_file_id)
        if full_run_id:
            selected_media, _ = full_run_review_media(manifest, episode, selected_media, full_run_id)
        elif joint_mode == "corrected":
            selected_media, _ = preferred_projection_review_media(manifest, episode, selected_media)
    except KeyError:
        raise HTTPException(404, "Episode 或视频流不存在")
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc))
    try:
        payload = preview_proxy_manager.submit(dataset_id, episode_id, selected_media, force_proxy) if start else preview_proxy_manager.status(dataset_id, episode_id, selected_media)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))
    result = dict(payload)
    if result.get("status") == "ready":
        run_query = f"&full_run_id={full_run_id}" if full_run_id else ""
        result["media_url"] = f"/api/datasets/{dataset_id}/episodes/{episode_id}/preview-media?media_file_id={selected_media.get('file_id') or ''}&joint_mode={joint_mode}{run_query}"
        if result.get("delivery") == "proxy":
            result["mapping_url"] = f"/api/datasets/{dataset_id}/episodes/{episode_id}/preview-mapping?media_file_id={selected_media.get('file_id') or ''}&joint_mode={joint_mode}{run_query}"
    result.pop("mapping_path", None)
    result["media_file_id"] = selected_media.get("file_id")
    result["stream_name"] = selected_media.get("stream_name")
    return result


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/preview-proxy")
def preview_proxy_status(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = None,
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
    full_run_id: str | None = None,
):
    return _preview_proxy_payload(dataset_id, episode_id, media_file_id, joint_mode=joint_mode, full_run_id=full_run_id)


@app.post("/api/datasets/{dataset_id}/episodes/{episode_id}/preview-proxy")
def prepare_preview_proxy(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = None,
    force_proxy: bool = False,
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
    full_run_id: str | None = None,
):
    return _preview_proxy_payload(dataset_id, episode_id, media_file_id, start=True, force_proxy=force_proxy, joint_mode=joint_mode, full_run_id=full_run_id)


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/preview-media")
def preview_media(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = None,
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
    full_run_id: str | None = None,
):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        selected_media = episode_media(episode, media_file_id)
        if full_run_id:
            selected_media, _ = full_run_review_media(manifest, episode, selected_media, full_run_id)
        elif joint_mode == "corrected":
            selected_media, _ = preferred_projection_review_media(manifest, episode, selected_media)
        path, media_type = preview_proxy_manager.media_path(dataset_id, episode_id, selected_media)
    except KeyError:
        raise HTTPException(404, "Episode 或视频流不存在")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/preview-mapping")
def preview_mapping(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = None,
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
    full_run_id: str | None = None,
):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        selected_media = episode_media(episode, media_file_id)
        if full_run_id:
            selected_media, _ = full_run_review_media(manifest, episode, selected_media, full_run_id)
        elif joint_mode == "corrected":
            selected_media, _ = preferred_projection_review_media(manifest, episode, selected_media)
        path = preview_proxy_manager.mapping_path(dataset_id, episode_id, selected_media)
    except KeyError:
        raise HTTPException(404, "Episode 或视频流不存在")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/joint-overlay/status")
def joint_overlay_availability(
    dataset_id: str,
    episode_id: str,
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        return joint_overlay_status(manifest, episode, joint_mode)
    except KeyError:
        raise HTTPException(404, "Episode 涓嶅瓨鍦?")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/pose-recovery/status")
def episode_pose_recovery_status(dataset_id: str, episode_id: str):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        return pose_recovery_status(dataset_id, manifest, episode)
    except KeyError:
        raise HTTPException(404, "Episode 涓嶅瓨鍦?")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/projection-correction/status")
def episode_projection_correction_status(dataset_id: str, episode_id: str):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        return projection_correction_status(dataset_id, manifest, episode)
    except KeyError:
        raise HTTPException(404, "Episode 不存在")


@app.post("/api/datasets/{dataset_id}/episodes/{episode_id}/pose-recovery")
def recover_episode_initial_pose(dataset_id: str, episode_id: str):
    try:
        get_episode(dataset_id, episode_id)
        return batch_analysis_jobs.submit(
            dataset_id,
            BatchAnalysisRequest(operation="pose_recovery", episode_ids=[episode_id]),
        )
    except KeyError:
        raise HTTPException(404, "Episode 涓嶅瓨鍦?")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/models/configure")
def configure_model(config: dict):
    try:
        if config.get("slot") == "hand_pose" or config.get("kind") in {"alicepose", "pose", "mediapipe"}:
            return registry.configure_hand_pose_async(HandPoseModelConfig.model_validate(config))
        if config.get("slot") == "vlm" or config.get("kind") == "qwen":
            validated = VLMModelConfig.model_validate(config)
            status = registry.configure_vlm(validated)
            _save_vlm_config(validated)
            return status
        return registry.configure_local(LocalModelConfig.model_validate(config))
    except Exception as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/models/upload")
def upload_model(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pt", ".onnx", ".engine"}:
        raise HTTPException(400, "仅支持 .pt、.onnx 或 .engine 模型文件")
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}-{Path(file.filename or 'model.pt').name}"
    target = models_dir / safe_name
    size = 0
    with target.open("wb") as output:
        while chunk := file.file.read(1024 * 1024):
            output.write(chunk)
            size += len(chunk)
    if size < 1024:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "模型文件为空或不完整")
    return {"ok": True, "path": str(target.relative_to(ROOT)), "size": size}


@app.get("/api/models/status")
def model_status():
    return registry.status()


@app.post("/api/datasets/{dataset_id}/episodes/{episode_id}/analyze")
def analyze(dataset_id: str, episode_id: str, request: AnalysisRequest):
    try:
        get_episode(dataset_id, episode_id)
        return jobs.submit(dataset_id, episode_id, request)
    except KeyError:
        raise HTTPException(404, "Episode 不存在")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/joint-overlay/frame")
def joint_overlay_frame(
    dataset_id: str,
    episode_id: str,
    index: int = Query(0, ge=0),
    media_file_id: str | None = None,
    joint_mode: str = Query("auto", pattern="^(auto|raw|corrected)$"),
    full_run_id: str | None = None,
):
    try:
        manifest_path = manifest_registry_path(dataset_id)
        manifest_revision = manifest_path.stat().st_mtime_ns if manifest_path.is_file() else 0
        manifest, episode = _cached_joint_context(dataset_id, episode_id, manifest_revision)
        selected_media = episode_media(episode, media_file_id)
        if full_run_id:
            selected_media, _ = full_run_review_media(manifest, episode, selected_media, full_run_id)
        elif joint_mode == "corrected":
            selected_media, _ = preferred_projection_review_media(manifest, episode, selected_media)
        width = int(selected_media.get("width") or episode.get("width") or 0)
        height = int(selected_media.get("height") or episode.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("视频尺寸无效")
        return joint_overlay_geometry(manifest, episode, index, width, height, selected_media, joint_mode)
    except KeyError:
        raise HTTPException(404, "Episode 或视频流不存在")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@lru_cache(maxsize=128)
def _cached_joint_context(dataset_id: str, episode_id: str, manifest_revision: int) -> tuple[dict, dict]:
    return get_episode(dataset_id, episode_id)


@app.post("/api/datasets/{dataset_id}/episodes/{episode_id}/annotate-behavior")
def annotate_behavior(dataset_id: str, episode_id: str, request: BehaviorAnnotationRequest):
    try:
        get_episode(dataset_id, episode_id)
        return behavior_jobs.submit(dataset_id, episode_id, request)
    except KeyError:
        raise HTTPException(404, "Episode 涓嶅瓨鍦?")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/datasets/{dataset_id}/analysis-jobs")
def submit_batch_analysis(dataset_id: str, request: BatchAnalysisRequest):
    try:
        return batch_analysis_jobs.submit(dataset_id, request)
    except KeyError:
        raise HTTPException(404, "Dataset or Episode not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/datasets/{dataset_id}/qwen-trim-jobs")
def submit_qwen_trim(dataset_id: str, request: QwenTrimRequest):
    try:
        return qwen_trim_jobs.submit(dataset_id, request)
    except KeyError:
        raise HTTPException(404, "Dataset or Episode not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/datasets/{dataset_id}/curation-jobs")
def submit_curation(dataset_id: str, request: CurationJobRequest):
    try:
        return curation_jobs.submit(dataset_id, request)
    except KeyError:
        raise HTTPException(404, "Dataset or Episode not found")
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/action-mappings/profiles")
def action_profiles():
    return {"items": action_mapping_profiles()}


@app.post("/api/datasets/{dataset_id}/action-jobs")
def submit_action_mapping(dataset_id: str, request: ActionMappingRequest):
    try:
        return action_mapping_jobs.submit(dataset_id, request)
    except KeyError:
        raise HTTPException(404, "Dataset 或 Episode 不存在")
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/action-jobs")
def list_action_mapping_jobs(dataset_id: str, active_only: bool = False):
    try:
        get_manifest(dataset_id)
        return {"items": action_mapping_jobs.list(dataset_id, active_only=active_only)}
    except KeyError:
        raise HTTPException(404, "Dataset 不存在")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/action-mapping")
def episode_action_mapping(dataset_id: str, episode_id: str, profile_id: str | None = None):
    try:
        get_episode(dataset_id, episode_id)
        payload = load_episode_action_mapping(dataset_id, episode_id, profile_id)
        if payload is None:
            raise HTTPException(404, "尚无 Action 映射结果")
        return payload
    except KeyError:
        raise HTTPException(404, "Dataset 或 Episode 不存在")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/curation-jobs")
def list_curation_jobs(dataset_id: str, active_only: bool = False):
    try:
        get_manifest(dataset_id)
        return curation_jobs.list(dataset_id, active_only=active_only)
    except KeyError:
        raise HTTPException(404, "Dataset 不存在")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/curation-preflight")
def episode_curation_preflight(dataset_id: str, episode_id: str, media_file_id: str | None = None):
    try:
        return curation_preflight(dataset_id, episode_id, media_file_id)
    except KeyError:
        raise HTTPException(404, "Dataset、Episode 或视频流不存在")
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/curation")
def episode_curation_result(dataset_id: str, episode_id: str, media_file_id: str | None = None, run_id: str | None = None):
    try:
        get_episode(dataset_id, episode_id)
        if run_id:
            bundle = load_full_run_episode_bundle(dataset_id, episode_id, media_file_id, run_id)
            payload = (bundle or {}).get("curation")
        else:
            payload = load_curation_report(dataset_id, episode_id, media_file_id)
        if payload is None:
            raise HTTPException(404, "尚无数据质量清洗报告")
        return payload
    except KeyError:
        raise HTTPException(404, "Episode 不存在")
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/full-run")
def episode_full_run_result(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = None,
    run_id: str | None = None,
):
    try:
        get_episode(dataset_id, episode_id)
        payload = load_full_run_episode_bundle(dataset_id, episode_id, media_file_id, run_id)
        if payload is None:
            raise HTTPException(404, "当前视频流尚无完整的 Full run")
        return payload
    except KeyError:
        raise HTTPException(404, "Episode 不存在")
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/qwen-trim-jobs")
def list_qwen_trim_jobs(dataset_id: str, active_only: bool = False):
    try:
        get_manifest(dataset_id)
        return {"items": qwen_trim_jobs.list(dataset_id, active_only=active_only)}
    except KeyError:
        raise HTTPException(404, "Dataset 不存在")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/qwen-action-trim")
def qwen_action_trim_result(dataset_id: str, episode_id: str):
    try:
        get_episode(dataset_id, episode_id)
        payload = load_qwen_action_trim(dataset_id, episode_id)
        if payload is None:
            raise HTTPException(404, "尚无 Qwen 有效/无效片段结果")
        return payload
    except KeyError:
        raise HTTPException(404, "Episode 不存在")


@app.post("/api/datasets/{dataset_id}/sensor-alignment")
def start_sensor_alignment(dataset_id: str, force: bool = False):
    try:
        return sensor_alignment_jobs.submit(dataset_id, force=force)
    except KeyError:
        raise HTTPException(404, "Dataset 不存在")
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/sensor-alignment")
def sensor_alignment_status(dataset_id: str):
    try:
        get_manifest(dataset_id)
        return sensor_alignment_jobs.status(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset 不存在")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/sensor-alignment")
def episode_sensor_alignment(dataset_id: str, episode_id: str):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        return load_sensor_alignment(manifest, episode_id) or scan_episode_sensor_alignment(manifest, episode)
    except KeyError:
        raise HTTPException(404, "Episode 不存在")
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/behavior-annotation")
def behavior_annotation(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
):
    try:
        manifest, episode = get_episode(dataset_id, episode_id)
        if run_id:
            bundle = load_full_run_episode_bundle(dataset_id, episode_id, media_file_id, run_id)
            payload = (bundle or {}).get("behavior")
            if payload is None:
                raise HTTPException(404, "该 Full run 没有 VLM 行为标注")
            return payload
        context = behavior_analysis_context(dataset_id, manifest, episode, media_file_id)
        status = behavior_annotation_status(
            dataset_id,
            manifest,
            episode,
            source_media_file_id=str(context["source_media"].get("file_id") or "") or None,
            analysis_media=context["analysis_media"],
            analysis_frame_ranges=context["analysis_frame_ranges"],
        )
        if not status.get("reusable"):
            raise HTTPException(404, "尚无 VLM 行为标注")
        return status["payload"]
    except KeyError:
        raise HTTPException(404, "Episode 涓嶅瓨鍦?")


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    for manager in (jobs, behavior_jobs, batch_analysis_jobs, qwen_trim_jobs, sensor_alignment_jobs, curation_jobs, action_mapping_jobs):
        try:
            return manager.get(job_id)
        except KeyError:
            continue
    raise HTTPException(404, "任务不存在")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    for manager in (jobs, behavior_jobs, batch_analysis_jobs, qwen_trim_jobs, sensor_alignment_jobs, curation_jobs, action_mapping_jobs):
        try:
            manager.get(job_id)
        except KeyError:
            continue
        return manager.cancel(job_id)
    raise HTTPException(404, "任务不存在")


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/annotations")
def annotations(dataset_id: str, episode_id: str):
    payload = load_annotations(dataset_id, episode_id)
    if payload is None:
        raise HTTPException(404, "尚无分析标注")
    return payload


@app.get("/api/datasets/{dataset_id}/changes")
def changes(dataset_id: str):
    try:
        return list_changes(dataset_id)
    except KeyError:
        raise HTTPException(404, "Dataset not found")


@app.post("/api/datasets/{dataset_id}/changes/apply")
def apply_dataset_changes(dataset_id: str, request: ApplyChangesRequest):
    if request.confirmation != "APPLY":
        raise HTTPException(400, "Explicit confirmation is required")
    try:
        return apply_changes(dataset_id, request.change_ids)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/invalid-index")
def invalid_frame_index(
    dataset_id: str,
    episode_id: str,
    frame: int | None = Query(None, ge=0),
):
    try:
        if frame is not None:
            invalid, result = is_frame_invalid(dataset_id, episode_id, frame)
            return {
                "dataset_id": dataset_id,
                "episode_id": episode_id,
                "frame": result["frame"],
                "frame_count": result["frame_count"],
                "invalid": invalid,
                "source": "alicePD_bitmap",
            }
        index = load_invalid_frame_index(dataset_id, episode_id)
        if index is None:
            raise KeyError(episode_id)
        return index
    except KeyError:
        raise HTTPException(404, "该 Episode 尚无无效帧索引")
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc))


def _annotation_base_for_media(dataset_id: str, episode_id: str, media: dict) -> dict:
    payload = load_annotations(dataset_id, episode_id)
    trim_payload = load_no_action_trim(dataset_id, episode_id)
    qwen_trim_payload = load_qwen_action_trim(dataset_id, episode_id)
    if payload is None or not (payload.get("manual_edits") or []):
        candidates = [item for item in (trim_payload, qwen_trim_payload) if item is not None]
        candidates = [
            item for item in candidates
            if not item.get("source_video", {}).get("file_id")
            or item.get("source_video", {}).get("file_id") == media.get("file_id")
        ]
        if candidates:
            payload = max(candidates, key=lambda item: str(item.get("created_at") or ""))
    return payload or {
        "schema": ALICE_ANNOTATION_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "summary": {},
        "segments": [],
        "samples": [],
    }


def _annotation_source_video(media: dict) -> dict:
    return {
        "file_id": media.get("file_id"),
        "stream_name": media.get("stream_name"),
        "relative_path": media.get("relative_path"),
        "frame_count": media.get("frame_count"),
        "fps": media.get("fps"),
        "width": media.get("width"),
        "height": media.get("height"),
    }


@app.patch("/api/datasets/{dataset_id}/episodes/{episode_id}/segments")
def update_segment(dataset_id: str, episode_id: str, update: SegmentUpdate):
    try:
        _, episode = get_episode(dataset_id, episode_id)
        media = episode_media(episode, update.media_file_id)
    except KeyError:
        raise HTTPException(404, "Episode 或视频流不存在")
    payload = _annotation_base_for_media(dataset_id, episode_id, media)
    try:
        payload = apply_segment_override(
            payload,
            update.model_dump(exclude={"media_file_id"}),
            int(media.get("frame_count", 0) or 0),
            float(media.get("fps", 30.0) or 30.0),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    payload["source_video"] = _annotation_source_video(media)
    save_annotations(dataset_id, episode_id, payload)
    return payload


@app.post("/api/datasets/{dataset_id}/episodes/{episode_id}/behavior-removals")
def remove_behavior_phase(dataset_id: str, episode_id: str, request: BehaviorPhaseRemovalRequest):
    try:
        _, episode = get_episode(dataset_id, episode_id)
        media = episode_media(episode, request.media_file_id)
    except KeyError:
        raise HTTPException(404, "Episode 或视频流不存在")
    behavior = load_behavior_annotation(dataset_id, episode_id, request.full_run_id)
    if behavior is None:
        raise HTTPException(409, "请先运行 VLM 行为标注，再按动作去除")
    behavior_media_id = (behavior.get("source_video") or {}).get("file_id")
    if behavior_media_id and media.get("file_id") and behavior_media_id != media.get("file_id"):
        raise HTTPException(409, "VLM 行为标注属于另一个视频流，请先为当前视频重新运行")
    payload = _annotation_base_for_media(dataset_id, episode_id, media)
    try:
        payload = apply_behavior_phase_exclusion(
            payload,
            behavior,
            request.phase_label,
            int(media.get("frame_count", 0) or 0),
            float(media.get("fps", 30.0) or 30.0),
            request.reason,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    payload["source_video"] = _annotation_source_video(media)
    save_annotations(dataset_id, episode_id, payload)
    return payload


@app.get("/api/datasets/{dataset_id}/episodes/{episode_id}/no-action-trim")
def no_action_trim_result(dataset_id: str, episode_id: str):
    try:
        get_episode(dataset_id, episode_id)
        payload = load_no_action_trim(dataset_id, episode_id)
        if payload is None:
            raise HTTPException(404, "尚无无动作剪切结果")
        return payload
    except KeyError:
        raise HTTPException(404, "Episode 不存在")


@app.get("/api/datasets/{dataset_id}/export.zip")
def download_export(dataset_id: str, include_media: bool = False):
    try:
        path = export_zip(dataset_id, include_media=include_media)
    except KeyError:
        raise HTTPException(404, "数据集不存在")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.post("/api/datasets/{dataset_id}/export-folder")
def write_export(dataset_id: str, request: ExportFolderRequest):
    try:
        path = export_dataset(dataset_id, Path(request.path), request.include_media)
    except KeyError:
        raise HTTPException(404, "数据集不存在")
    except OSError as exc:
        raise HTTPException(422, f"导出失败: {exc}")
    return {"ok": True, "path": str(path)}


@app.post("/api/datasets/{dataset_id}/export-folder-dialog")
def write_export_with_dialog(dataset_id: str, include_media: bool = False):
    selected = choose_folder()
    if selected is None:
        return {"cancelled": True, "path": None}
    try:
        path = export_dataset(dataset_id, selected, include_media)
    except KeyError:
        raise HTTPException(404, "数据集不存在")
    except OSError as exc:
        raise HTTPException(422, f"导出失败: {exc}")
    return {"cancelled": False, "path": str(path)}


@app.get("/")
def frontend():
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store, max-age=0"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
