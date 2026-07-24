from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".vla_lens"
STATE_FILE = RUNTIME / "server.json"
STDOUT_LOG = RUNTIME / "server.out.log"
STDERR_LOG = RUNTIME / "server.err.log"
REQUIRED_MODULES = (
    "fastapi", "uvicorn", "cv2", "ultralytics", "httpx", "h5py",
    "pyarrow", "imageio_ffmpeg", "scipy", "torch",
)


def _print(payload: Any, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif isinstance(payload, str):
        print(payload)
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _request_json(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 3.0) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            detail = str(error_payload.get("detail") or error_payload)
        except (OSError, ValueError, json.JSONDecodeError):
            detail = str(exc.reason or exc)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _health(url: str, timeout: float = 2.0) -> dict | None:
    try:
        payload = _request_json(f"{url.rstrip('/')}/api/health", timeout=timeout)
        return payload if payload.get("ok") and payload.get("service") == "vla-lens" else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _load_state() -> dict | None:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_state(payload: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_FILE)


def _state_health(state: dict | None) -> dict | None:
    if not state or not state.get("url"):
        return None
    health = _health(str(state["url"]))
    expected = state.get("instance_id")
    if health and expected and health.get("instance_id") != expected:
        return None
    return health


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind((host, port))
            return True
        except OSError:
            return False


def _tail(path: Path, count: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])
    except OSError:
        return ""


def command_serve(args: argparse.Namespace) -> int:
    if args.instance_id:
        os.environ["VLA_INSTANCE_ID"] = args.instance_id
    if args.no_model:
        os.environ["VLA_SKIP_MODEL_AUTOLOAD"] = "1"
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("缺少 uvicorn；请先运行 run.sh --setup 或 run.ps1 -Setup") from exc
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=not args.quiet,
        log_level=args.log_level,
    )
    return 0


def command_start(args: argparse.Namespace) -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    health = _state_health(state)
    if health:
        result = {"status": "already_running", "url": state["url"], "pid": state.get("pid"), "models": health.get("models")}
        _print(result, args.json)
        if args.browser:
            webbrowser.open(str(state["url"]))
        return 0

    selected_port = None
    for port in range(args.port, args.max_port + 1):
        url = f"http://{args.host}:{port}"
        existing = _health(url)
        if existing:
            result = {"status": "already_running", "url": f"{url}/", "pid": None, "models": existing.get("models")}
            _print(result, args.json)
            if args.browser:
                webbrowser.open(f"{url}/")
            return 0
        if _port_available(args.host, port):
            selected_port = port
            break
    if selected_port is None:
        raise RuntimeError(f"端口 {args.port}-{args.max_port} 均不可用")

    instance_id = uuid.uuid4().hex
    url = f"http://{args.host}:{selected_port}/"
    env = dict(os.environ)
    env["VLA_INSTANCE_ID"] = instance_id
    command = [sys.executable, "-m", "app.cli", "serve", "--host", args.host, "--port", str(selected_port), "--instance-id", instance_id]
    if args.no_model:
        command.append("--no-model")
    if args.quiet:
        command.append("--quiet")
    creationflags = 0
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        popen_options["start_new_session"] = True
    started_at = datetime.now(timezone.utc).isoformat()
    with STDOUT_LOG.open("a", encoding="utf-8") as stdout, STDERR_LOG.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n--- alice start {started_at} instance={instance_id} ---\n")
        stderr.write(f"\n--- alice start {started_at} instance={instance_id} ---\n")
        stdout.flush()
        stderr.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            **popen_options,
        )
    state = {
        "schema": "alice/service-state/v2",
        "pid": process.pid,
        "port": selected_port,
        "url": url,
        "instance_id": instance_id,
        "python": sys.executable,
        "started_at": started_at,
    }
    _save_state(state)

    deadline = time.monotonic() + args.wait
    health = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"服务启动失败（退出码 {process.returncode}）\n{_tail(STDERR_LOG)}")
        health = _state_health(state)
        if health:
            break
        time.sleep(0.2)
    if not health:
        raise RuntimeError(f"服务在 {args.wait:g} 秒内未就绪\n{_tail(STDERR_LOG)}")

    state["service_pid"] = int(health.get("pid") or process.pid)
    _save_state(state)

    result = {"status": "started", "url": url, "pid": process.pid, "models": health.get("models")}
    _print(result, args.json)
    if args.browser:
        webbrowser.open(url)
    return 0


def command_status(args: argparse.Namespace) -> int:
    state = _load_state()
    health = _state_health(state)
    result = {
        "running": bool(health),
        "url": state.get("url") if state else None,
        "pid": state.get("pid") if state else None,
        "health": health,
    }
    _print(result, args.json)
    return 0 if health else 1


def command_stop(args: argparse.Namespace) -> int:
    state = _load_state()
    if not state or not state.get("pid"):
        _print({"status": "not_running"}, args.json)
        return 0
    pid = int(state["pid"])
    health = _state_health(state)
    if health is None and not args.force:
        STATE_FILE.unlink(missing_ok=True)
        _print({"status": "stale_state_removed", "pid": pid}, args.json)
        return 0
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if completed.returncode not in {0, 128} and _health(str(state.get("url") or ""), timeout=0.3):
                raise OSError(completed.stderr.strip() or completed.stdout.strip() or f"taskkill exit {completed.returncode}")
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise RuntimeError(f"无法停止进程 {pid}: {exc}") from exc
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and _health(str(state.get("url") or ""), timeout=0.3):
        time.sleep(0.2)
    STATE_FILE.unlink(missing_ok=True)
    _print({"status": "stopped", "pid": pid}, args.json)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    model = ROOT / "yoloe-26x-seg.pt"
    encoder = ROOT / "mobileclip2_b.ts"
    config = RUNTIME / "vlm-config.json"
    result = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "dependencies": "ok" if not missing else f"missing: {', '.join(missing)}",
        "yoloe_model": model.is_file() and model.stat().st_size > 1_000_000,
        "text_encoder": encoder.is_file() and encoder.stat().st_size > 100_000_000,
        "qwen_config": config.is_file(),
        "runtime_writable": os.access(RUNTIME if RUNTIME.exists() else ROOT, os.W_OK),
    }
    _print(result, args.json)
    return 0 if not missing and result["yoloe_model"] else 1


def command_open(args: argparse.Namespace) -> int:
    source = str(Path(args.path).expanduser().resolve())
    payload = _request_json(
        f"{args.url.rstrip('/')}/api/datasets/open-path",
        method="POST",
        payload={"path": source, "name": args.name},
        timeout=args.timeout,
    )
    _print(payload, args.json)
    return 0


def command_datasets(args: argparse.Namespace) -> int:
    payload = _request_json(f"{args.url.rstrip('/')}/api/datasets", timeout=args.timeout)
    _print(payload, args.json)
    return 0


def command_schema(args: argparse.Namespace) -> int:
    suffix = "/analyze-schema" if args.analyze else "/schema"
    method = "POST" if args.analyze else "GET"
    payload = _request_json(f"{args.url.rstrip('/')}/api/datasets/{args.dataset_id}{suffix}", method=method, timeout=args.timeout)
    _print(payload, args.json)
    return 0


def _robot_profile_id(value: str) -> str:
    from .action_mapping import ACTION_MAPPING_PROFILES

    profile_id = str(value).strip()
    if profile_id not in ACTION_MAPPING_PROFILES:
        available = ", ".join(ACTION_MAPPING_PROFILES)
        raise argparse.ArgumentTypeError(f"未知机器人类型 {profile_id!r}；可选值: {available}")
    return profile_id


def command_robots(args: argparse.Namespace) -> int:
    from .action_mapping import action_mapping_profiles

    profiles = action_mapping_profiles()
    if args.json:
        _print({"items": profiles}, True)
        return 0
    print("可选机器人类型：")
    for profile in profiles:
        sides = "双臂" if int(profile.get("sides") or 0) == 2 else "单臂"
        requirement = "需标定/IK" if profile.get("requires_ik") else "无需 IK"
        print(f"  {profile['id']:<24} {int(profile.get('action_dim') or 0):>2}D  {sides:<4}  {requirement:<8}  {profile['name']}")
    return 0


def _full_service_url(value: str) -> str:
    if value != "auto":
        return value.rstrip("/")
    state = _load_state()
    health = _state_health(state)
    if state and health:
        return str(state["url"]).rstrip("/")
    for port in range(8000, 8011):
        candidate = f"http://127.0.0.1:{port}"
        if _health(candidate, timeout=0.5):
            return candidate
    raise RuntimeError("Alice service is not running; start it with sh run.sh or sh full.sh")


def _resolve_full_dataset(args: argparse.Namespace, url: str) -> dict:
    candidate = Path(args.target).expanduser()
    if args.path or candidate.exists():
        return _request_json(
            f"{url}/api/datasets/open-path",
            method="POST",
            payload={
                "path": str(candidate.resolve()),
                "name": args.name,
                "analyze_schema": not args.skip_schema,
            },
            timeout=args.request_timeout,
        )
    dataset_id = urllib.parse.quote(args.target, safe="")
    return _request_json(f"{url}/api/datasets/{dataset_id}", timeout=args.request_timeout)


def _select_full_episodes(manifest: dict, patterns: list[str], all_episodes: bool) -> list[dict]:
    episodes = list(manifest.get("episodes") or [])
    if all_episodes == bool(patterns):
        raise RuntimeError("Choose exactly one scope: --all or one or more --episode values")
    if all_episodes:
        return episodes
    selected: list[dict] = []
    unmatched: list[str] = []
    for pattern_group in patterns:
        for pattern in (item.strip() for item in pattern_group.split(",")):
            if not pattern:
                continue
            matches = [
                episode
                for episode in episodes
                if fnmatch.fnmatchcase(str(episode.get("id") or ""), pattern)
                or fnmatch.fnmatchcase(str(episode.get("name") or ""), pattern)
            ]
            if not matches:
                unmatched.append(pattern)
            for episode in matches:
                if all(str(existing.get("id")) != str(episode.get("id")) for existing in selected):
                    selected.append(episode)
    if unmatched:
        raise RuntimeError(f"Episode not found: {', '.join(unmatched)}")
    if not selected:
        raise RuntimeError("No Episodes were selected")
    return selected


def _full_media_file_id(episode: dict, requested: str | None) -> str:
    streams = list(episode.get("media_streams") or [])
    if requested:
        wanted = requested.casefold()
        matches = [
            item
            for item in streams
            if wanted in {
                str(item.get("file_id") or "").casefold(),
                str(item.get("stream_name") or "").casefold(),
                str(item.get("relative_path") or "").casefold(),
                Path(str(item.get("relative_path") or "")).name.casefold(),
            }
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{episode.get('name') or episode.get('id')}: media selector {requested!r} matched {len(matches)} streams"
            )
        return str(matches[0]["file_id"])
    preferred = str(episode.get("primary_media_file_id") or "")
    if preferred and any(str(item.get("file_id") or "") == preferred for item in streams):
        return preferred
    if len(streams) == 1 and streams[0].get("file_id"):
        return str(streams[0]["file_id"])
    raise RuntimeError(
        f"{episode.get('name') or episode.get('id')}: choose one of the media streams with --media"
    )


def _balanced_full_shards(episodes: list[dict], count: int) -> list[list[dict]]:
    shard_count = max(1, min(int(count), len(episodes)))
    shards: list[list[dict]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    ordered = sorted(
        episodes,
        key=lambda item: (-int(item.get("frame_count") or 0), str(item.get("id") or "")),
    )
    for episode in ordered:
        index = min(range(shard_count), key=lambda value: (loads[value], value))
        shards[index].append(episode)
        loads[index] += max(1, int(episode.get("frame_count") or 0))
    return [shard for shard in shards if shard]


def _validate_full_runtime(args: argparse.Namespace, health: dict) -> dict:
    runtime = ((health.get("runtime") or {}).get("full_pipeline") or {})
    workers = int(runtime.get("workers") or 0)
    gpu_devices = [str(item) for item in runtime.get("gpu_devices") or []]
    if args.expect_workers is not None and workers != args.expect_workers:
        raise RuntimeError(
            f"Running service has {workers or 'unknown'} Full workers, expected {args.expect_workers}. "
            "Stop it with sh stop.sh and rerun full.sh so the GPU configuration takes effect."
        )
    expected_gpus = [item.strip() for item in (args.expect_gpus or "").split(",") if item.strip()]
    if expected_gpus and gpu_devices != expected_gpus:
        raise RuntimeError(
            f"Running service GPU list is {gpu_devices or 'empty'}, expected {expected_gpus}. "
            "Stop it with sh stop.sh and rerun full.sh."
        )
    return runtime


def _cancel_full_jobs(url: str, jobs: list[dict], timeout: float) -> None:
    for job in jobs:
        if str(job.get("status") or "") in {"complete", "failed", "cancelled"}:
            continue
        try:
            _request_json(f"{url}/api/jobs/{job['id']}/cancel", method="POST", timeout=timeout)
        except (OSError, RuntimeError, urllib.error.URLError):
            pass


def command_full(args: argparse.Namespace) -> int:
    url = _full_service_url(args.url)
    health = _health(url, timeout=min(args.request_timeout, 10.0))
    if not health:
        raise RuntimeError(f"Alice service is not healthy at {url}")
    runtime = _validate_full_runtime(args, health)
    manifest = _resolve_full_dataset(args, url)
    if not manifest.get("id") or not isinstance(manifest.get("episodes"), list):
        raise RuntimeError("The selected path did not resolve to one dataset")
    episodes = _select_full_episodes(manifest, args.episode or [], args.all)
    media_file_ids = {
        str(episode["id"]): _full_media_file_id(episode, args.media)
        for episode in episodes
    }
    shards = _balanced_full_shards(episodes, args.parallel)
    submitted: list[dict] = []
    try:
        for shard in shards:
            episode_ids = [str(item["id"]) for item in shard]
            payload = {
                "episode_ids": episode_ids,
                "media_file_ids": {episode_id: media_file_ids[episode_id] for episode_id in episode_ids},
                "full_pipeline": True,
                "force_vlm": bool(args.force_vlm),
                "vlm_sample_count": int(args.vlm_samples),
            }
            if args.action_profile:
                payload.update({
                    "full_action_profile_id": args.action_profile,
                    "full_action_source_hand": args.source_hand,
                    "full_action_coordinate_frame": args.coordinate_frame,
                    "full_action_horizon_frames": int(args.horizon_frames),
                })
            job = _request_json(
                f"{url}/api/datasets/{urllib.parse.quote(str(manifest['id']), safe='')}/curation-jobs",
                method="POST",
                payload=payload,
                timeout=args.request_timeout,
            )
            submitted.append(job)
    except Exception:
        _cancel_full_jobs(url, submitted, args.request_timeout)
        raise

    launch = {
        "status": "submitted",
        "dataset_id": manifest["id"],
        "dataset_name": manifest.get("name"),
        "episode_count": len(episodes),
        "shard_count": len(submitted),
        "job_ids": [job["id"] for job in submitted],
        "robot_profile": args.action_profile,
        "action_generation": bool(args.action_profile),
        "runtime": runtime,
    }
    if args.detach:
        _print(launch, args.json)
        return 0
    if not args.json:
        print(
            f"Full started: {len(episodes)} Episodes, {len(submitted)} parallel jobs, "
            f"workers={runtime.get('workers') or 'unknown'}, GPUs={runtime.get('gpu_devices') or 'CPU'}, "
            f"robot={args.action_profile or 'existing/none'}"
        )

    terminal = {"complete", "failed", "cancelled"}
    started = time.monotonic()
    current = submitted
    last_line = ""
    try:
        while True:
            current = [
                _request_json(f"{url}/api/jobs/{job['id']}", timeout=args.request_timeout)
                for job in current
            ]
            completed_episodes = sum(int(job.get("completed_count") or 0) for job in current)
            progress = sum(float(job.get("progress") or 0.0) * int(job.get("episode_count") or 1) for job in current)
            progress /= max(1, sum(int(job.get("episode_count") or 1) for job in current))
            active = next((job for job in current if str(job.get("status") or "") not in terminal), current[-1])
            line = f"[{progress:6.2f}%] {completed_episodes}/{len(episodes)} Episodes | {active.get('message') or active.get('status')}"
            if not args.json and line != last_line:
                print(line[:180].ljust(180), end="\r", flush=True)
                last_line = line
            if all(str(job.get("status") or "") in terminal for job in current):
                break
            if args.wait_timeout > 0 and time.monotonic() - started > args.wait_timeout:
                raise RuntimeError(f"Full did not finish within {args.wait_timeout:g} seconds")
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        _cancel_full_jobs(url, current, args.request_timeout)
        if not args.json:
            print("\nCancellation requested for all Full jobs.")
        return 130
    except Exception:
        _cancel_full_jobs(url, current, args.request_timeout)
        raise
    finally:
        if not args.json and last_line:
            print()

    failure_count = sum(int((job.get("result") or {}).get("failure_count") or 0) for job in current)
    pair_count = sum(int((job.get("result") or {}).get("pair_count") or 0) for job in current)
    output_roots = sorted({
        str((job.get("result") or {}).get("output_root"))
        for job in current
        if (job.get("result") or {}).get("output_root")
    })
    failed_jobs = [job["id"] for job in current if str(job.get("status") or "") != "complete"]
    result = {
        **launch,
        "status": "failed" if failed_jobs or failure_count else "complete",
        "pair_count": pair_count,
        "failure_count": failure_count,
        "failed_job_ids": failed_jobs,
        "output_roots": output_roots,
        "jobs": current,
    }
    _print(result, args.json)
    return 1 if failed_jobs or failure_count else 0


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="输出 JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alice", description="Alice Blue command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="前台运行 API 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--instance-id")
    serve.add_argument("--no-model", action="store_true", help="跳过本地模型自动加载")
    serve.add_argument("--quiet", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=command_serve)

    start = subparsers.add_parser("start", help="后台启动服务")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--max-port", type=int, default=8010)
    start.add_argument("--wait", type=float, default=30)
    start.add_argument("--browser", action="store_true")
    start.add_argument("--no-model", action="store_true")
    start.add_argument("--quiet", action="store_true")
    _add_output_options(start)
    start.set_defaults(handler=command_start)

    status = subparsers.add_parser("status", help="显示服务与模型状态")
    _add_output_options(status)
    status.set_defaults(handler=command_status)

    stop = subparsers.add_parser("stop", help="停止本项目启动的服务")
    stop.add_argument("--force", action="store_true")
    _add_output_options(stop)
    stop.set_defaults(handler=command_stop)

    doctor = subparsers.add_parser("doctor", help="检查 Python、依赖、模型和配置")
    _add_output_options(doctor)
    doctor.set_defaults(handler=command_doctor)

    full = subparsers.add_parser("full", help="Run the complete dataset pipeline and follow its progress")
    full.add_argument("target", help="Registered dataset ID or local dataset path")
    full.add_argument("--path", action="store_true", help="Always interpret target as a local path")
    full.add_argument("--name", help="Dataset name when opening a local path")
    scope = full.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true", help="Process every Episode")
    scope.add_argument("--episode", action="append", help="Episode ID/name/glob; repeat or comma-separate values")
    full.add_argument("--media", help="Media file ID, stream name, relative path, or basename")
    full.add_argument("--parallel", type=int, default=1, choices=range(1, 33), metavar="N", help="Balanced parallel Full jobs")
    full.add_argument("--force-vlm", action="store_true", help="Do not reuse matching VLM annotations")
    full.add_argument("--vlm-samples", type=int, default=18, choices=range(6, 25), metavar="N")
    full.add_argument(
        "--robot",
        "--action-profile",
        dest="action_profile",
        type=_robot_profile_id,
        metavar="TYPE",
        help="可选机器人类型；不传则不生成派生 Action。使用 `sh full.sh --robots` 查看列表",
    )
    full.add_argument("--source-hand", choices=("left", "right"), default="right")
    full.add_argument("--coordinate-frame", choices=("camera", "world"), default="camera")
    full.add_argument("--horizon-frames", type=int, default=3, choices=range(1, 31), metavar="N")
    full.add_argument("--skip-schema", action="store_true", help="Do not run schema understanding when opening a path")
    full.add_argument("--detach", action="store_true", help="Submit jobs and return immediately")
    full.add_argument("--poll-seconds", type=float, default=2.0)
    full.add_argument("--wait-timeout", type=float, default=0.0, help="Overall wait limit; 0 means unlimited")
    full.add_argument("--request-timeout", type=float, default=300.0)
    full.add_argument("--url", default="auto", help="Service URL or 'auto' to use .vla_lens/server.json")
    full.add_argument("--expect-workers", type=int)
    full.add_argument("--expect-gpus")
    _add_output_options(full)
    full.set_defaults(handler=command_full)

    robots = subparsers.add_parser("robots", help="列出 Full 可生成 Action 的机器人类型")
    _add_output_options(robots)
    robots.set_defaults(handler=command_robots)

    for name, help_text, handler in (
        ("datasets", "列出已登记数据集", command_datasets),
        ("open", "从命令行打开数据集目录", command_open),
        ("schema", "查看或执行 Schema 理解", command_schema),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--url", default="http://127.0.0.1:8000")
        command.add_argument("--timeout", type=float, default=300)
        if name == "open":
            command.add_argument("path")
            command.add_argument("--name")
        elif name == "schema":
            command.add_argument("dataset_id")
            command.add_argument("--analyze", action="store_true")
        _add_output_options(command)
        command.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"alice: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
