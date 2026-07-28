(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const state = { dataset: null, datasetCollection: null, datasetCache: new Map(), datasetLoadToken: 0, episode: null, media: null, file: null, fileById: new Map(), episodeById: new Map(), treeGroups: new Map(), treeExpanded: new Set(), treeCollapsed: new Set(), treeAutoCollapse: false, treeSearchTimer: null, treeSelection: null, annotations: null, curation: null, curationStageFilter: null, curationPreflightToken: 0, reviewSelectionToken: 0, changes: null, models: null, pendingSourcePaths: new Set(), analysisOperation: null, analysisJobs: new Map(), analysisMonitors: new Set(), analysisReturnFocus: null, actionProfiles: [], actionMapping: null, actionMappingReturnFocus: null, sensorAlignmentToken: 0, sensorAlignmentJob: null, frame: 0, zoom: 1, playing: false, jointOverlay: false, jointOverlayAvailable: false, yoloOverlay: false, yoloOverlayReport: null, yoloOverlaySamples: [], yoloOverlayLoadToken: 0, yoloOverlaySampleFrame: -1, timer: null, playbackToken: 0, playbackStartedAt: 0, playbackStartFrame: 0, frameImageToken: 0, frameImageResolve: null, frameImageTimeout: null, frameAbortController: null, frameObjectUrl: null, nativePreview: null, previewPollToken: 0, nativeFrameCallback: null, nativePresentedFrame: -1, nativePresentedFrames: 0, nativeMediaTime: null, jointGeometryAbortController: null, jointGeometryLastAt: 0, jointGeometryFrame: -1, jointGeometryDesiredFrame: -1, jointGeometryInFlightFrame: -1, jointGeometryPendingFrame: null, jointGeometryRequestToken: 0, jointGeometryTimer: null, view: "dashboard", treeMode: "episode", frameData: { fileId: null, field: null, index: 0, count: 0, mode: null, follow: true, requestToken: 0, timer: null, pendingIndex: 0, lastRequestAt: 0 }, h5Compare: { index: 0, field: null, count: 0, requestToken: 0, timer: null, pendingIndex: 0 } };
  state.jointIndices = false;
  state.jointGeometryCurrent = null;
  const esc = value => { const node = document.createElement("span"); node.textContent = String(value ?? ""); return node.innerHTML; };
  const escAttr = value => esc(value).replaceAll('"', "&quot;");
  const qualityDisplayText = value => String(value ?? "").replaceAll("\u8bba\u6587\u5f0f", "").replaceAll("\u8bba\u6587", "").replace(/\s{2,}/g, " ").trim();
  const fmtBytes = value => { let n = Number(value || 0), unit = "B"; for (const next of ["KB", "MB", "GB", "TB"]) { if (n < 1024) break; n /= 1024; unit = next; } return `${n < 10 && unit !== "B" ? n.toFixed(1) : Math.round(n)} ${unit}`; };
  const fmtTime = value => { const seconds = Math.max(0, Number(value || 0)); return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(3).padStart(6, "0")}`; };
  const api = async (url, options = {}) => { const response = await fetch(url, { cache: "no-store", ...options }); let data = null; try { data = await response.json(); } catch (_) {} if (!response.ok) throw new Error(data?.detail || data?.error || `请求失败 (${response.status})`); return data; };
  const toast = (message, type = "") => { const box = $("#toast"); box.textContent = message; box.className = `toast show ${type}`; clearTimeout(box._timer); box._timer = setTimeout(() => box.classList.remove("show"), 3000); };
  const setStatus = message => { $("#statusText").textContent = message || "就绪"; };
  const setProgress = (value, message) => { const strip = $("#globalProgress"); strip.classList.remove("indeterminate"); strip.classList.toggle("hidden", value == null); $("#progressFill").style.width = `${Math.max(0, Math.min(100, Number(value || 0)))}%`; $("#progressText").textContent = message || "处理中"; };
  function beginQwenProgress(mode = "manual") {
    const startedAt = Date.now(), strip = $("#globalProgress"), busy = $("#schemaBusy"), overview = $(".schema-overview");
    strip.classList.remove("hidden"); strip.classList.add("indeterminate"); $("#progressFill").style.width = "32%"; busy.classList.remove("hidden"); overview.classList.add("running"); $("#schemaStatusBadge").textContent = "分析中"; $("#analyzeSchemaButton").disabled = true; lucide.createIcons();
    const automatic = mode === "auto", update = () => {
      const seconds = Math.floor((Date.now() - startedAt) / 1000), elapsed = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
      const detail = seconds < 12 ? (automatic ? "扫描完成后将自动发送真实结构清单" : "正在发送已压缩的真实结构清单") : seconds < 60 ? "正在识别左右手、视觉、关节与传感器关系" : seconds < 120 ? "正在生成结构关联并审计 Episode，服务仍在运行" : "Qwen 仍在推理，大型数据集可能需要数分钟，请继续等待";
      $("#schemaBusyDetail").textContent = detail; $("#schemaBusyElapsed").textContent = `已等待 ${elapsed}`; $("#progressText").textContent = `${automatic ? "自动 Qwen 理解仍在运行" : "Qwen 请求仍在运行"} · ${elapsed}`; setStatus(`${automatic ? "正在自动理解数据结构" : "Qwen 正在分析"} · 已等待 ${elapsed} · 请求未中断`);
    };
    update(); const timer = setInterval(update, 1000);
    return () => { clearInterval(timer); strip.classList.remove("indeterminate"); strip.classList.add("hidden"); busy.classList.add("hidden"); overview.classList.remove("running"); $("#analyzeSchemaButton").disabled = !state.dataset; };
  }
  function beginDatasetLoadProgress() {
    if (state.models?.vlm?.configured) return beginQwenProgress("auto");
    setProgress(12, "正在扫描数据集并生成本地结构清单");
    setStatus("正在扫描数据集 · Qwen 未配置，暂不自动理解");
    return () => setProgress(null);
  }

  function beginLazyDatasetProgress() {
    setProgress(12, "正在按需建立所选数据集的本地索引");
    setStatus("正在读取所选数据集 · 不调用 Qwen");
    return () => setProgress(null);
  }

  function renderDatasetSelector() {
    const selector = $("#datasetSelect"), collection = state.datasetCollection, items = collection?.datasets || [];
    selector.classList.toggle("hidden", !items.length);
    if (!items.length) { selector.innerHTML = ""; selector.disabled = true; return; }
    selector.innerHTML = items.map(item => {
      const status = item.status === "loading" ? "加载中" : item.status === "loaded" ? `${Number(item.file_count || 0).toLocaleString()} 文件` : "未加载";
      return `<option value="${escAttr(item.key)}">${esc(item.name)} · ${status}</option>`;
    }).join("");
    selector.value = collection.activeKey || items[0].key;
    selector.disabled = Boolean(collection.loadingKey);
    selector.title = `${collection.rootPath} · ${items.length} 个独立数据集`;
  }

  function rememberCollectionDataset(manifest) {
    const collection = state.datasetCollection;
    if (!collection?.activeKey) return;
    const item = collection.datasets.find(candidate => candidate.key === collection.activeKey);
    if (!item) return;
    item.status = "loaded"; item.dataset_id = manifest.id; item.file_count = Number(manifest.file_count || manifest.files?.length || 0); item.episode_count = Number(manifest.episode_count || 0);
    state.datasetCache.delete(item.key); state.datasetCache.set(item.key, manifest);
    while (state.datasetCache.size > 3) state.datasetCache.delete(state.datasetCache.keys().next().value);
    renderDatasetSelector();
  }

  function installDatasetCollection(data) {
    state.datasetLoadToken += 1;
    state.datasetCollection = { rootPath: data.root_path, activeKey: null, loadingKey: null, datasets: (data.datasets || []).map(item => ({ ...item })) };
    state.datasetCache = new Map(); state.dataset = null; setEnabled(false);
    $("#titleDataset").textContent = `${data.dataset_count || data.datasets?.length || 0} 个数据集`;
    $("#datasetName").textContent = "选择数据集"; $("#datasetPath").textContent = data.root_path;
    $("#fileTree").innerHTML = '<div class="empty"><i data-lucide="database"></i><b>数据集尚未加载</b><span>选择上方数据集后按需建立索引</span></div>';
    renderDatasetSelector(); lucide.createIcons();
  }

  async function loadCollectionDataset(key) {
    const collection = state.datasetCollection, item = collection?.datasets.find(candidate => candidate.key === key);
    if (!collection || !item || collection.loadingKey) return;
    const token = ++state.datasetLoadToken, finishProgress = beginLazyDatasetProgress();
    collection.loadingKey = key; collection.activeKey = key; item.status = "loading"; renderDatasetSelector();
    try {
      let manifest = state.datasetCache.get(key);
      if (!manifest && item.dataset_id) {
        try { manifest = await api(`/api/datasets/${encodeURIComponent(item.dataset_id)}`); } catch (_) { item.dataset_id = null; }
      }
      if (!manifest) manifest = await api("/api/datasets/open-path", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: item.path, name: item.name, analyze_schema: false }) });
      if (token !== state.datasetLoadToken) return;
      item.status = "loaded"; collection.loadingKey = null; collection.activeKey = key; renderDataset(manifest);
      const schemaStatus = manifest.schema_profile?.status;
      toast(schemaStatus === "completed" ? `已加载 ${manifest.name} · 格式已理解` : `已加载 ${manifest.name}`, schemaStatus === "error" ? "error" : "");
    } catch (error) {
      if (token === state.datasetLoadToken) { item.status = "unloaded"; collection.loadingKey = null; renderDatasetSelector(); setStatus(error.message); toast(error.message, "error"); }
    } finally { finishProgress(); }
  }

  function renderSensorAlignmentStatus(job) {
    const node = $("#sensorSyncStatus"), status = job?.status || "queued", result = job?.result || {}, summary = result.summary || result;
    if (status === "idle") { node.className = "sensor-sync-status"; node.textContent = "SYNC: 按需检测"; node.title = job.message || "进入分析流程时再检测传感器时间对齐"; return; }
    node.className = `sensor-sync-status ${status === "complete" ? (Number(summary.conflict_count || 0) ? "warning" : "ready") : status === "failed" ? "failed" : "running"}`;
    if (status === "complete") {
      const streams = Number(summary.stream_count ?? result.stream_count ?? 0), timestamp = Number(summary.timestamp_aligned_count || 0), scaled = Number(summary.rate_multiplier_count ?? summary.multiplier_aligned_count ?? summary.scaled_stream_count ?? 0), conflicts = Number(summary.conflict_count || 0);
      node.textContent = `SYNC: ${streams} 流 · ${timestamp} 时间戳 · ${scaled} 倍率${conflicts ? ` · ${conflicts} 冲突` : ""}`;
      node.title = summary.message || `传感器时钟检测完成；${timestamp} 路按时间戳映射，${scaled} 路按 Hz 倍率映射${conflicts ? `；${conflicts} 路声明频率存在冲突，已保留原始信息` : ""}`;
      return;
    }
    if (status === "failed") { node.textContent = "SYNC: 检测失败"; node.title = job.error || job.message || "传感器时钟检测失败"; return; }
    const completed = Number(job.completed_count || 0), total = Number(job.episode_count || 0);
    node.textContent = `SYNC: 检测中${total ? ` ${completed}/${total}` : ""}`;
    node.title = job.message || "正在后台检测视频与各传感器 Hz";
  }
  async function startSensorAlignment(datasetId) {
    const token = ++state.sensorAlignmentToken;
    renderSensorAlignmentStatus({ status: "queued", message: "等待传感器时钟检测" });
    try {
      let job = await api(`/api/datasets/${encodeURIComponent(datasetId)}/sensor-alignment`, { method: "POST" });
      while (token === state.sensorAlignmentToken && state.dataset?.id === datasetId) {
        state.sensorAlignmentJob = job; renderSensorAlignmentStatus(job);
        if (["complete", "failed"].includes(job.status)) return;
        await new Promise(resolve => setTimeout(resolve, 750));
        job = await api(`/api/jobs/${encodeURIComponent(job.id)}`);
      }
    } catch (error) {
      if (token !== state.sensorAlignmentToken || state.dataset?.id !== datasetId) return;
      renderSensorAlignmentStatus({ status: "failed", error: error.message });
    }
  }

  function setEnabled(enabled) {
    ["#refreshButton", "#treeMode", "#treeSearch", "#exportFolderButton", "#downloadExport", "#analyzeSchemaButton", "#reviewChangesButton"].forEach(id => { $(id).disabled = !enabled; });
    $("#excludeFileButton").disabled = !enabled || !state.treeSelection?.fileIds?.length;
    const hasEpisodes = Boolean(enabled && state.dataset?.episodes?.length);
    ["#curationPipelineButton", "#fullPipelineButton", "#actionMappingButton", "#videoSmoothButton", "#poseRecoveryButton", "#behaviorAnnotateButton", "#noActionTrimButton", "#qwenTrimButton"].forEach(id => { $(id).disabled = !hasEpisodes; });
    ["#manualRangeButton", "#playButton", "#transportPlay", "#prevFrame", "#nextFrame", "#frameSlider", "#zoomIn", "#zoomOut"].forEach(id => { $(id).disabled = !(enabled && Boolean(state.episode)); });
  }

  function renderDataset(manifest) {
    resetNativePreview(); resetYoloOverlay(true); resetJointIndices();
    state.reviewSelectionToken += 1;
    state.dataset = manifest; state.episode = null; state.media = null; state.file = null; state.fileById = new Map((manifest.files || []).map(file => [file.id, file])); state.episodeById = new Map((manifest.episodes || []).map(episode => [episode.id, episode])); state.treeGroups = new Map(); state.treeExpanded = new Set(); state.treeCollapsed = new Set(); state.treeAutoCollapse = Number(manifest.file_count || manifest.files?.length || 0) > 500; state.treeSelection = null; state.annotations = null; state.curation = null; state.curationStageFilter = null; state.actionMapping = null; state.changes = null; state.pendingSourcePaths = new Set(); state.frame = 0; state.jointOverlay = false; state.jointOverlayAvailable = false; $("#jointOverlayButton").classList.remove("active"); $("#jointOverlayButton").disabled = true; $("#yoloOverlayButton").disabled = true; $("#excludeFileButton").disabled = true; rememberCollectionDataset(manifest); updateTreeSelectionUI();
    hideFrameDataInspector(); clearBehaviorAnnotation(); clearCurationReport(); clearActionMappingResult();
    $("#titleDataset").textContent = manifest.name; $("#datasetName").textContent = manifest.name; $("#datasetPath").textContent = manifest.root_path;
    const sidecar = $("#datasetSidecar"); sidecar.classList.toggle("hidden", !manifest.sidecar_path); sidecar.textContent = manifest.sidecar_path ? `派生目录 ${manifest.sidecar_path} · 已隔离 ${Number(manifest.auxiliary_file_count || 0)} 个辅助文件 · 已移出 ${Number(manifest.excluded_file_count || 0)} 个条目` : "";
    $("#episodeCount").textContent = Number(manifest.episode_count || 0).toLocaleString(); $("#dashboardFiles").textContent = Number(manifest.file_count || manifest.files?.length || 0).toLocaleString(); $("#frameCount").textContent = Number(manifest.frame_count || 0).toLocaleString(); $("#dashboardSize").textContent = fmtBytes(manifest.total_size);
    $("#fileCount").textContent = `${manifest.file_count || manifest.files?.length || 0} 个文件`; $("#totalSize").textContent = fmtBytes(manifest.total_size);
    $("#workspaceMeta").textContent = `${manifest.episode_count || 0} Episodes · ${manifest.file_count || 0} files`;
    renderBreakdown(manifest.type_counts || {}); renderModels(); renderResolvedTree(); renderSchema(manifest.schema_profile); setEnabled(true); showView("dashboard"); loadChangeCatalog(); renderSensorAlignmentStatus({ status: "idle", message: "传感器对齐改为分析时按需启动" }); restoreAnalysisJobs(manifest.id);
    setStatus("数据集已打开，源目录保持只读");
  }

  function updateChangeApplyState() {
    const checked = $$("#changeList input[data-change-id]:checked").length > 0;
    $("#confirmApplyChanges").disabled = !checked || !$("#changeConfirm").checked;
  }
  function renderChangeCatalog(catalog) {
    state.changes = catalog || { pending_count: 0, applied_count: 0, items: [] };
    state.pendingSourcePaths = new Set((state.changes.items || []).filter(item => item.status === "pending").flatMap(item => item.source_paths || []));
    const pending = Number(state.changes.pending_count || 0), applied = Number(state.changes.applied_count || 0);
    const rerun = Number(state.changes.requires_rerun_count ?? (state.changes.items || []).filter(item => item.status === "pending" && item.requires_rerun).length);
    const runnable = Number(state.changes.runnable_pending_count ?? Math.max(0, pending - rerun));
    const pendingLabel = rerun ? `${runnable} 项可应用 · ${rerun} 项需重新运行` : `${pending} 项待应用`;
    $("#changeStatusBadge").textContent = pending ? `${pendingLabel} · ${applied} 项已应用` : (applied ? `${applied} 项已应用` : "无待应用更改");
    $("#applyChangesButton").disabled = runnable === 0;
    $("#reviewChangesButton").disabled = !state.dataset;
    $("#changeSummary").textContent = pending ? `${runnable} 项待确认${rerun ? `；${rerun} 项需重新运行` : ""}；${applied} 项已生成应用快照` : (applied ? `没有待确认更改；已应用 ${applied} 项` : "当前没有更改记录");
    $("#changeList").innerHTML = (state.changes.items || []).map(item => {
      const isPending = item.status === "pending";
      const requiresRerun = isPending && item.kind === "paper_curation" && Boolean(item.requires_rerun);
      const selectable = isPending && !requiresRerun;
      const summary = item.summary || {};
      const detail = item.kind === "vlm_behavior" ? `${summary.task_label || "other"} · ${Number(summary.target_count || 0)} 个主要目标` : item.kind === "paper_curation" ? `${Number(summary.invalid_frame_count || 0).toLocaleString()} 异常帧 · ${Number(summary.review_frame_count || 0).toLocaleString()} 待审 · ${Number(summary.stage_completed_count || 0)}/8 阶段` : item.kind === "episode_annotation" ? `${summary.segment_count || 0} 个片段 · ${summary.invalid_count || 0} 个无效片段` : item.kind === "no_action_trim" ? `${Number(summary.valid_frame_count || 0).toLocaleString()} 有效帧 · ${Number(summary.invalid_frame_count || 0).toLocaleString()} 无动作帧` : item.kind === "qwen_action_trim" ? `${Number(summary.valid_frame_count || 0).toLocaleString()} 有效帧 · ${Number(summary.invalid_frame_count || 0).toLocaleString()} Qwen 无效帧` : item.kind === "video_smoothing" ? `${summary.stream_name || "video"} · ${Number(summary.frame_count || 0).toLocaleString()} 帧平滑` : item.kind === "dataset_exclusion" ? `本次移出 ${Number(summary.excluded_count || 0)} 个 · 累计 ${Number(summary.total_excluded || 0)} 个` : `${summary.recovered_frame_count || 0} 帧恢复建议`;
      const statusLabel = requiresRerun ? "需重新运行" : selectable ? "待应用" : "已应用";
      const rowClass = requiresRerun ? "requires-rerun" : isPending ? "pending" : "applied";
      const checkbox = `<input type="checkbox" data-change-id="${escAttr(item.id)}" ${selectable ? "checked" : "disabled"}>`;
      const versionNote = requiresRerun ? ` · pipeline v${Number(item.pipeline_version ?? 1)}，算法已升级` : "";
      const title = item.kind === "paper_curation" ? qualityDisplayText(item.title || item.key) : (item.title || item.key);
      return `<label class="change-row ${rowClass}">${checkbox}<span class="change-row-main"><b>${esc(title)}</b><small>${esc(detail)}${esc(versionNote)} · revision ${Number(item.revision || 0)}</small></span><span class="change-row-status">${statusLabel}</span></label>`;
    }).join("") || '<div class="empty-copy">暂无更改记录</div>';
    $("#changeConfirm").checked = false; updateChangeApplyState();
    $$("#changeList input[data-change-id]").forEach(input => input.addEventListener("change", updateChangeApplyState));
    if (state.dataset) renderResolvedTree();
  }
  async function loadChangeCatalog() {
    if (!state.dataset) return;
    try { renderChangeCatalog(await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/changes`)); }
    catch (error) { $("#changeStatusBadge").textContent = "更改记录不可用"; $("#applyChangesButton").disabled = true; toast(error.message, "error"); }
  }
  async function openChangeModal() {
    if (!state.dataset) return;
    await loadChangeCatalog(); $("#changeModal").classList.remove("hidden"); lucide.createIcons();
  }
  function closeChangeModal() { $("#changeModal").classList.add("hidden"); }
  async function applySelectedChanges() {
    if (!state.dataset) return;
    const changeIds = $$("#changeList input[data-change-id]:checked").map(input => input.dataset.changeId);
    if (!changeIds.length || !$("#changeConfirm").checked) return;
    $("#confirmApplyChanges").disabled = true; setProgress(8, "正在生成应用快照");
    try {
      const result = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/changes/apply`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ change_ids: changeIds, confirmation: "APPLY" }) });
      renderChangeCatalog(result.catalog); closeChangeModal(); setStatus(`已应用 ${changeIds.length} 项更改；源数据文件仍保持只读`); toast(`已应用 ${changeIds.length} 项更改`, "");
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { setProgress(null); updateChangeApplyState(); }
  }

  function renderBreakdown(counts) { const total = Math.max(1, Object.values(counts).reduce((a, b) => a + Number(b || 0), 0)); const labels = { video: "视频", image: "图像", structured: "结构化", metadata: "元数据", text: "文本", file: "其他" }; $("#typeBreakdown").innerHTML = Object.entries(counts).map(([kind, count]) => `<div class="breakdown-row"><b>${labels[kind] || kind}</b><span><i style="width:${Number(count) / total * 100}%"></i></span><em>${count}</em></div>`).join("") || `<div class="empty-copy">没有文件</div>`; }

  function renderModels() {
    api("/api/models/status").then(status => { state.models = status; const local = status.local || {}, vlm = status.vlm || {}, localState = local.loaded ? "READY" : local.loading ? "LOADING" : "OFF", localDetail = local.loaded ? `已加载 · ${esc(local.device)} · ${esc(local.warmup_ms)} ms` : local.loading ? "后台加载中，文件管理器可先使用" : esc(local.error || "未加载"); $("#modelSummary").innerHTML = `<div class="model-item"><i data-lucide="scan"></i><div><b>${esc(local.family || local.kind || "本地模型")}</b><small>${localDetail}</small></div><span class="badge ${local.loaded ? "ready" : ""}">${localState}</span></div><div class="model-item"><i data-lucide="sparkles"></i><div><b>Qwen-VLM</b><small>${vlm.configured ? `${esc(vlm.model)} · 已配置` : esc(vlm.error || "未配置")}</small></div><span class="badge ${vlm.configured ? "ready" : ""}">${vlm.configured ? "READY" : "OFF"}</span></div>`; lucide.createIcons(); $("#statusModel").textContent = local.loaded ? `MODEL: ${local.family || local.kind}` : local.loading ? "MODEL: LOADING" : "MODEL: --"; updateYoloOverlayAvailability(); if (local.loading) setTimeout(renderModels, 1200); }).catch(() => {});
  }

  function fileIcon(file) { return file.kind === "video" ? "video" : file.kind === "structured" ? "database" : file.kind === "metadata" ? "file-text" : file.kind === "image" ? "image" : "file"; }
  function row(file, indent = 0) { const pending = state.pendingSourcePaths.has(file.relative_path), active = state.treeSelection?.kind === "file" && state.treeSelection.id === file.id; return `<div class="tree-row file ${pending ? "pending-change" : ""} ${active ? "active" : ""}" data-file-id="${escAttr(file.id)}" style="padding-left:${10 + indent * 17}px"><i class="tree-icon" data-lucide="${fileIcon(file)}"></i><span class="tree-label" title="${escAttr(pending ? `${file.relative_path} · 有待应用更改` : file.relative_path)}">${esc(file.name)}</span><span class="tree-meta">${pending ? "待应用" : fmtBytes(file.size_bytes)}</span></div>`; }
  function group(label, children, key, meta = "", options = {}) {
    const pending = children.some(file => state.pendingSourcePaths.has(file.relative_path));
    const selectable = Boolean(options.selectable), active = selectable && state.treeSelection?.kind === "episode" && state.treeSelection.id === key;
    const selectionData = selectable ? ` data-select-kind="episode" data-episode-id="${escAttr(options.episodeId || "")}"` : "";
    const query = $("#treeSearch").value.trim(), expanded = Boolean(query || state.treeExpanded.has(key) || (!state.treeAutoCollapse && !state.treeCollapsed.has(key)));
    return `<div class="tree-node"><div class="tree-row group ${selectable ? "selectable" : ""} ${pending ? "pending-change" : ""} ${active ? "active" : ""}" data-group="${escAttr(key)}" data-label="${escAttr(label)}"${selectionData}><button class="tree-toggle" title="展开或收起">${expanded ? "▾" : "›"}</button><i class="tree-icon" data-lucide="${selectable ? "folder-kanban" : "folder"}"></i><b class="tree-label" title="${escAttr(label)}">${esc(label)}</b><span class="tree-meta">${pending ? "有更改" : (meta || children.length)}</span></div><div class="tree-children ${expanded ? "" : "collapsed"}">${expanded ? children.map(file => row(file, 1)).join("") : ""}</div></div>`;
  }
  function renderTree() {
    const files = state.dataset?.files || [], mode = state.treeMode; const groups = new Map();
    for (const file of files) { let key, label, episodeId = null; if (mode === "episode") { key = file.episode_id || `unassigned:${file.episode_key}`; const episode = state.episodeById.get(file.episode_id); label = episode?.name || file.episode_key; episodeId = episode?.id || null; } else if (mode === "category") { key = file.category || "other"; label = { vision: "视觉 Vision", sensor: "传感器 Joint / Tactile", metadata: "元数据", other: "其他" }[key] || key; } else { key = file.parent || "."; label = key === "." ? state.dataset.name : key; } if (!groups.has(key)) groups.set(key, { label, files: [], episodeId }); groups.get(key).files.push(file); }
    state.treeGroups = groups; const query = $("#treeSearch").value.trim().toLowerCase(); let html = ""; for (const [key, value] of groups) { const children = value.files.filter(file => !query || `${file.name} ${file.relative_path}`.toLowerCase().includes(query)); if (children.length || !query) html += group(value.label, children, key, "", { selectable: mode === "episode" && Boolean(value.episodeId), episodeId: value.episodeId }); } $("#fileTree").innerHTML = html || `<div class="empty"><i data-lucide="search-x"></i><b>没有匹配文件</b></div>`; lucide.createIcons(); bindTree(); updateTreeSelectionUI(); }
  function renderResolvedTree() {
    const files = state.dataset?.files || [], mode = state.treeMode, query = $("#treeSearch").value.trim().toLowerCase();
    if (mode !== "episode") { $("#episodeWarning").classList.add("hidden"); renderTree(); return; }
    const byId = state.fileById;
    const resolution = state.dataset?.episode_resolution || {};
    const groups = [];
    for (const item of resolution.groups || []) {
      groups.push({ key: item.group_id, label: item.label || item.group_id, files: (item.file_ids || []).map(id => byId.get(id)).filter(Boolean), meta: `${item.source === "qwen" ? "AI" : "LOCAL"} · ${(item.file_ids || []).length}`, selectable: true, episodeId: item.playable_episode_id || "" });
    }
    const shared = (resolution.shared_file_ids || []).map(id => byId.get(id)).filter(Boolean);
    const unassigned = (resolution.unassigned_file_ids || []).map(id => byId.get(id)).filter(Boolean);
    if (shared.length) groups.push({ key: "shared", label: "共享 / 数据集级文件", files: shared, meta: String(shared.length) });
    if (unassigned.length) groups.push({ key: "unassigned", label: "未分配（不是 Episode）", files: unassigned, meta: String(unassigned.length) });
    if (!(resolution.groups || []).length && files.length) groups.push({ key: "unassigned", label: "未分配（请重新扫描）", files, meta: String(files.length) });
    const warning = $("#episodeWarning");
    const requiresApi = Boolean(resolution.requires_api || unassigned.length);
    warning.classList.toggle("hidden", !requiresApi && !resolution.ai_confirmed);
    warning.classList.toggle("confirmed", Boolean(resolution.ai_confirmed));
    warning.textContent = resolution.ai_confirmed
      ? `${resolution.model || "Qwen"} 已审计 Episode 归属；仍有 ${unassigned.length} 个文件未分配。`
      : requiresApi ? `${unassigned.length} 个文件缺少可靠 EP 证据。需要配置 Qwen API 后点击“再次运行”，否则保持未分配。` : "";
    state.treeGroups = new Map(groups.map(item => [item.key, item])); let html = "";
    for (const item of groups) {
      const children = item.files.filter(file => !query || `${file.name} ${file.relative_path}`.toLowerCase().includes(query));
      if (children.length || !query) html += group(item.label, children, item.key, item.meta, { selectable: item.selectable, episodeId: item.episodeId });
    }
    $("#fileTree").innerHTML = html || `<div class="empty"><i data-lucide="search-x"></i><b>没有匹配文件</b></div>`;
    lucide.createIcons(); bindTree(); updateTreeSelectionUI();
  }
  function updateTreeSelectionUI() {
    const selection = state.treeSelection, button = $("#excludeFileButton"), label = $("#excludeFileLabel"), hint = $("#treeSelectionHint");
    const available = Boolean(state.dataset && selection?.fileIds?.length);
    button.disabled = !available;
    label.textContent = selection?.kind === "episode" ? "移出 EP" : selection?.kind === "file" ? "移出文件" : "移出";
    hint.textContent = selection ? `${selection.kind === "episode" ? "EP" : "文件"} · ${selection.label}${selection.kind === "episode" ? ` · ${selection.fileIds.length} 项` : ""}` : "选择文件或 Episode";
    hint.title = selection?.label || "";
    $$(".tree-row.group").forEach(node => node.classList.toggle("active", selection?.kind === "episode" && node.dataset.group === selection.id));
    $$(".tree-row.file").forEach(node => node.classList.toggle("active", selection?.kind === "file" && node.dataset.fileId === selection.id));
  }
  async function selectEpisodeGroup(node) {
    const fileIds = (state.treeGroups.get(node.dataset.group)?.files || []).map(file => file.id);
    if (!fileIds.length) return;
    state.file = null;
    state.treeSelection = { kind: "episode", id: node.dataset.group, label: node.dataset.label || node.dataset.group, fileIds, episodeId: node.dataset.episodeId || null };
    updateTreeSelectionUI(); hideFrameDataInspector();
    $("#fileDetail").classList.add("hidden"); $("#fileEmpty").classList.remove("hidden");
    if (state.treeSelection.episodeId) await selectEpisode(state.treeSelection.episodeId);
    else setStatus(`已选择 ${state.treeSelection.label} · ${fileIds.length} 个条目`);
  }
  function toggleTreeGroup(node) { const key = node.dataset.group, expanded = state.treeExpanded.has(key) || (!state.treeAutoCollapse && !state.treeCollapsed.has(key)); if (expanded) { state.treeExpanded.delete(key); state.treeCollapsed.add(key); } else { state.treeCollapsed.delete(key); state.treeExpanded.add(key); } renderResolvedTree(); }
  function bindTree() {
    const tree = $("#fileTree"); if (tree.dataset.bound) return; tree.dataset.bound = "1";
    tree.addEventListener("click", event => { const toggle = event.target.closest(".tree-toggle"), groupNode = event.target.closest(".tree-row.group"), fileNode = event.target.closest(".tree-row.file"); if (toggle && groupNode) { event.stopPropagation(); toggleTreeGroup(groupNode); return; } if (groupNode) { groupNode.dataset.selectKind === "episode" ? selectEpisodeGroup(groupNode) : toggleTreeGroup(groupNode); return; } if (fileNode) selectFile(fileNode.dataset.fileId); });
  }

  async function selectFile(fileId) { const file = state.fileById.get(fileId); if (!file) return; state.file = file; state.treeSelection = { kind: "file", id: file.id, label: file.name, fileIds: [file.id], episodeId: file.episode_id || null }; updateTreeSelectionUI(); hideFrameDataInspector(); try { const detail = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/files/${encodeURIComponent(fileId)}`); renderFileDetail(detail); const playable = state.dataset?.episode_resolution?.file_episode_assignments?.[file.id] || file.episode_id; if (file.kind === "video") { if (playable) await selectEpisode(playable, file.id); return; } if (file.kind === "image") { if (playable) await selectEpisode(playable); return; } await openFilePreview(file, detail); } catch (error) { toast(error.message, "error"); } }
  function renderFileDetail(detail) { $("#fileEmpty").classList.add("hidden"); const fields = detail.fields || []; $("#fileDetail").classList.remove("hidden"); $("#fileDetail").innerHTML = `<div class="detail-title">${esc(detail.name)}</div><dl class="detail-grid"><dt>相对路径</dt><dd>${esc(detail.relative_path)}</dd><dt>类型</dt><dd>${esc(detail.kind)} / ${esc(detail.category)}</dd><dt>大小</dt><dd>${fmtBytes(detail.size_bytes)}</dd><dt>Episode</dt><dd>${esc(detail.episode?.name || detail.episode_key || "未归属")}</dd><dt>修改时间</dt><dd>${esc(detail.modified_at)}</dd></dl>${fields.length ? `<div class="field-list"><b>结构字段 (${fields.length})</b>${fields.slice(0, 80).map(field => `<div class="field-row">${esc(field.key || "$")} · ${esc(field.dtype || "")} · ${esc(JSON.stringify(field.shape || ""))}</div>`).join("")}</div>` : ""}`; }

  function stopPlayback() { state.playing = false; state.playbackToken += 1; clearTimeout(state.timer); state.timer = null; const video = $("#videoPlayer"); if (video && !video.paused) video.pause(); if (video?.cancelVideoFrameCallback && state.nativeFrameCallback != null) video.cancelVideoFrameCallback(state.nativeFrameCallback); state.nativeFrameCallback = null; if (state.nativePreview && video?.readyState >= 2 && state.episode) { updateNativeVideoFrame(); if (state.jointOverlay && state.nativePresentedFrame >= 0) requestJointGeometry(state.nativePresentedFrame, true); } if (state.frameAbortController) state.frameAbortController.abort(); state.frameAbortController = null; if (state.frameImageResolve) state.frameImageResolve(false); $("#transportPlay").innerHTML = '<i data-lucide="play"></i>'; $("#playButton").innerHTML = '<i data-lucide="play"></i>'; lucide.createIcons(); }
  function showMediaReview() { $("#reviewToolbar").classList.remove("hidden"); $("#mediaReview").classList.remove("hidden"); $("#filePreview").classList.add("hidden"); }
  async function openFilePreview(file, detail = null, field = null) {
    const hasEpisodes = Boolean(state.dataset?.episodes?.length); $("#curationPipelineButton").disabled = !hasEpisodes; $("#fullPipelineButton").disabled = !hasEpisodes; $("#actionMappingButton").disabled = !hasEpisodes; $("#videoSmoothButton").disabled = !hasEpisodes; $("#poseRecoveryButton").disabled = !hasEpisodes; $("#behaviorAnnotateButton").disabled = !hasEpisodes; $("#noActionTrimButton").disabled = !hasEpisodes; $("#qwenTrimButton").disabled = !hasEpisodes;
    stopPlayback(); showView("review"); $("#reviewToolbar").classList.add("hidden"); $("#mediaReview").classList.add("hidden"); $("#filePreview").classList.remove("hidden");
    $("#manualRangeButton").disabled = true; $("#workspaceTitle").textContent = `文件 / ${file.name}`; $("#workspaceMeta").textContent = file.relative_path; $("#statusEpisode").textContent = `FILE: ${file.name}`;
    $("#previewTitle").textContent = file.name; $("#previewPath").textContent = file.relative_path; $("#previewMode").textContent = "LOADING"; $("#previewControls").classList.add("hidden"); $("#previewNotice").classList.add("hidden"); $("#previewContent").innerHTML = '<div class="empty-copy">正在读取安全预览…</div>';
    try {
      const query = field ? `?field=${encodeURIComponent(field)}` : "";
      const preview = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/files/${encodeURIComponent(file.id)}/preview${query}`);
      if (state.file?.id !== file.id) return;
      renderUnifiedPreview(preview); configureFrameDataInspector(preview);
    } catch (error) { if (state.file?.id === file.id) $("#previewContent").innerHTML = `<div class="preview-error">${esc(error.message)}</div>`; }
  }
  function previewValue(value) { return value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value); }

  function hideFrameDataInspector() {
    state.frameData.requestToken += 1; clearTimeout(state.frameData.timer); state.frameData.timer = null; state.frameData.fileId = null; state.frameData.field = null; state.frameData.count = 0; state.frameData.mode = null;
    $("#frameDataInspector").classList.add("hidden");
  }
  function configureFrameDataInspector(preview) {
    const supported = new Set(["hdf5", "json", "numpy", "table", "parquet", "text"]);
    if (!state.file || !supported.has(preview.mode)) { hideFrameDataInspector(); return; }
    const changedFile = state.frameData.fileId !== state.file.id;
    state.frameData.fileId = state.file.id; state.frameData.mode = preview.mode;
    state.frameData.field = preview.selected_field || (changedFile ? null : state.frameData.field);
    state.frameData.index = state.frameData.follow && state.episode ? state.frame : (changedFile ? 0 : state.frameData.index);
    $("#frameDataInspector").classList.remove("hidden"); $("#frameDataValue").textContent = "正在读取真实帧值…";
    loadFrameData(state.frameData.index, state.frameData.field);
  }
  function scheduleFrameData(index, force = false) {
    if (!state.frameData.fileId || (!force && !state.frameData.follow) || $("#frameDataInspector").classList.contains("hidden")) return;
    state.frameData.pendingIndex = Math.max(0, Number(index) || 0);
    if (state.frameData.timer) return;
    const delay = Math.max(0, 140 - (Date.now() - state.frameData.lastRequestAt));
    state.frameData.timer = setTimeout(() => { state.frameData.timer = null; loadFrameData(state.frameData.pendingIndex, state.frameData.field); }, delay);
  }
  async function loadFrameData(index, field = null) {
    const fileId = state.frameData.fileId, datasetId = state.dataset?.id;
    if (!fileId || !datasetId) return;
    const token = ++state.frameData.requestToken; state.frameData.lastRequestAt = Date.now();
    const query = new URLSearchParams({ index: String(Math.max(0, Number(index) || 0)) }); if (field) query.set("field", field);
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(datasetId)}/files/${encodeURIComponent(fileId)}/frame?${query}`);
      if (token !== state.frameData.requestToken || state.file?.id !== fileId) return;
      renderFrameData(payload);
    } catch (error) {
      if (token === state.frameData.requestToken) { $("#frameDataMeta").textContent = "逐帧读取失败"; $("#frameDataValue").textContent = error.message; }
    }
  }
  function renderFrameData(payload) {
    if (payload.mode === "error") { $("#frameDataMeta").textContent = "该文件暂不能逐帧读取"; $("#frameDataValue").textContent = payload.error || "读取失败"; return; }
    const fields = payload.fields || [], selector = $("#frameDataField");
    state.frameData.mode = payload.mode; state.frameData.field = payload.field; state.frameData.index = Number(payload.frame_index || 0); state.frameData.count = Math.max(1, Number(payload.frame_count || 1));
    selector.innerHTML = fields.map(item => `<option value="${esc(item.path)}"${item.path === payload.field ? " selected" : ""}>${esc(item.path)} · ${esc(JSON.stringify(item.shape || []))} · ${esc(item.dtype || "")}</option>`).join("");
    selector.disabled = fields.length < 2; $("#frameDataIndex").value = state.frameData.index; $("#frameDataIndex").max = state.frameData.count - 1; $("#frameDataSlider").value = state.frameData.index; $("#frameDataSlider").max = state.frameData.count - 1; $("#frameDataCount").textContent = `/ ${state.frameData.count}`;
    const fullShape = JSON.stringify(payload.full_value_shape || []), shownShape = JSON.stringify(payload.value_shape || []), truncated = payload.truncated ? " · 当前显示为部分单帧内容" : "";
    $("#frameDataMeta").textContent = `${payload.dtype || "value"} · 原始 ${fullShape} · 当前 ${shownShape}${truncated}`;
    $("#frameDataValue").textContent = JSON.stringify(payload.value, null, 2) ?? String(payload.value ?? "");
    if (["hdf5", "parquet"].includes(payload.mode) && $("#h5Compare")) scheduleH5Comparison(state.frameData.index);
  }
  function renderH5CompareFrame(side, payload) {
    const title = $(`#h5Compare${side}Title`), meta = $(`#h5Compare${side}Meta`), value = $(`#h5Compare${side}Value`);
    if (!title || !meta || !value) return;
    title.textContent = `第 ${Number(payload.frame_index || 0)} 帧`;
    meta.textContent = `${payload.dtype || "value"} · Shape ${JSON.stringify(payload.full_value_shape || [])}${payload.truncated ? ` · 当前显示 ${JSON.stringify(payload.value_shape || [])}` : ""}`;
    value.textContent = JSON.stringify(payload.value, null, 2) ?? String(payload.value ?? "");
  }
  function collectH5Leaves(value, path = "$", result = []) {
    if (Array.isArray(value)) {
      value.forEach((item, index) => collectH5Leaves(item, `${path}[${index}]`, result));
    } else if (value && typeof value === "object") {
      Object.entries(value).forEach(([key, item]) => collectH5Leaves(item, `${path}.${key}`, result));
    } else {
      result.push([path, value]);
    }
    return result;
  }
  function formatH5DiffNumber(value) {
    if (!Number.isFinite(value)) return String(value);
    const absolute = Math.abs(value);
    if (absolute !== 0 && (absolute >= 1e12 || absolute < 1e-4)) return value.toExponential(5);
    return Number(value.toFixed(6)).toString();
  }
  function renderH5Difference(left, right) {
    const status = $("#h5DiffStatus"), changedNode = $("#h5DiffChanged"), meanNode = $("#h5DiffMean"), maxNode = $("#h5DiffMax"), preview = $("#h5DiffPreview"), scope = $("#h5DiffScope"), panel = $("#h5Diff");
    if (!status || !changedNode || !meanNode || !maxNode || !preview || !panel) return;
    const leftMap = new Map(collectH5Leaves(left.value)), rightMap = new Map(collectH5Leaves(right.value));
    const keys = [...new Set([...leftMap.keys(), ...rightMap.keys()])];
    const differences = [], numericDeltas = [];
    for (const key of keys) {
      const hasLeft = leftMap.has(key), hasRight = rightMap.has(key), before = leftMap.get(key), after = rightMap.get(key);
      if (hasLeft && hasRight && typeof before === "number" && typeof after === "number" && Number.isFinite(before) && Number.isFinite(after)) {
        const delta = after - before; numericDeltas.push(Math.abs(delta));
        if (delta !== 0) differences.push({ key, before, after, delta, magnitude: Math.abs(delta), numeric: true });
      } else if (!hasLeft || !hasRight || JSON.stringify(before) !== JSON.stringify(after)) {
        differences.push({ key, before: hasLeft ? before : "<缺失>", after: hasRight ? after : "<缺失>", magnitude: Number.POSITIVE_INFINITY, numeric: false });
      }
    }
    differences.sort((a, b) => b.magnitude - a.magnitude);
    const max = numericDeltas.length ? Math.max(...numericDeltas) : 0;
    const mean = numericDeltas.length ? numericDeltas.reduce((sum, value) => sum + value, 0) / numericDeltas.length : 0;
    const total = keys.length || 1, identical = differences.length === 0;
    status.textContent = identical ? "两帧一致" : "检测到变化";
    changedNode.textContent = `${differences.length} / ${total}`;
    meanNode.textContent = numericDeltas.length ? formatH5DiffNumber(mean) : "n/a";
    maxNode.textContent = numericDeltas.length ? formatH5DiffNumber(max) : "n/a";
    panel.classList.toggle("same", identical); panel.classList.toggle("changed", !identical);
    scope.textContent = left.truncated || right.truncated ? "基于当前返回的单帧内容比较（单帧内容有截断）" : `已比较 ${total} 个真实单帧叶子值`;
    preview.textContent = identical ? "未发现差异。" : differences.slice(0, 24).map(item => item.numeric
      ? `${item.key}: ${formatH5DiffNumber(item.before)} → ${formatH5DiffNumber(item.after)}   Δ ${item.delta >= 0 ? "+" : ""}${formatH5DiffNumber(item.delta)}`
      : `${item.key}: ${JSON.stringify(item.before)} → ${JSON.stringify(item.after)}`).join("\n") + (differences.length > 24 ? `\n… 另有 ${differences.length - 24} 项差异` : "");
  }
  async function loadH5Comparison(index, field = null) {
    const fileId = state.file?.id, datasetId = state.dataset?.id;
    if (!fileId || !datasetId || !$("#h5Compare")) return;
    const max = Math.max(0, Number($("#h5CompareIndex")?.max || 0)), resolved = Math.max(0, Math.min(Number(index) || 0, max));
    state.h5Compare.index = resolved; state.h5Compare.field = field || state.h5Compare.field; const token = ++state.h5Compare.requestToken;
    $("#h5CompareIndex").value = resolved; $("#h5CompareSlider").value = resolved; $("#h5CompareLeftValue").textContent = "正在读取第 n 帧…"; $("#h5CompareRightValue").textContent = "正在读取第 n+1 帧…"; if ($("#h5DiffPreview")) $("#h5DiffPreview").textContent = "正在比较真实帧值…";
    const makeUrl = frame => { const query = new URLSearchParams({ index: String(frame) }); if (state.h5Compare.field) query.set("field", state.h5Compare.field); return `/api/datasets/${encodeURIComponent(datasetId)}/files/${encodeURIComponent(fileId)}/frame?${query}`; };
    try {
      const [left, right] = await Promise.all([api(makeUrl(resolved)), api(makeUrl(resolved + 1))]);
      if (token !== state.h5Compare.requestToken || state.file?.id !== fileId || !$("#h5Compare")) return;
      renderH5CompareFrame("Left", left); renderH5CompareFrame("Right", right); renderH5Difference(left, right);
    } catch (error) {
      if (token === state.h5Compare.requestToken && $("#h5Compare")) { $("#h5CompareLeftValue").textContent = error.message; $("#h5CompareRightValue").textContent = error.message; }
    }
  }
  function scheduleH5Comparison(index, force = false) {
    if (!$("#h5Compare") || state.h5Compare.field == null) return;
    state.h5Compare.pendingIndex = Math.max(0, Number(index) || 0);
    if (state.h5Compare.timer && !force) return;
    clearTimeout(state.h5Compare.timer); state.h5Compare.timer = setTimeout(() => { state.h5Compare.timer = null; loadH5Comparison(state.h5Compare.pendingIndex, state.h5Compare.field); }, 90);
  }
  function bindH5Comparison(selected) {
    const count = Math.max(1, Number(selected?.shape?.[0] || 1)), max = Math.max(0, count - 2);
    state.h5Compare.field = selected?.path || null; state.h5Compare.count = count;
    const initial = Math.max(0, Math.min(state.frameData.fileId === state.file?.id ? state.frameData.index : 0, max));
    $("#h5CompareIndex").max = max; $("#h5CompareSlider").max = max; $("#h5CompareCount").textContent = `/ ${count - 1}`;
    const change = value => {
      const resolved = Math.max(0, Math.min(Number(value) || 0, max));
      state.h5Compare.index = resolved;
      $("#h5CompareIndex").value = resolved;
      $("#h5CompareSlider").value = resolved;
      loadH5Comparison(resolved, state.h5Compare.field);
      if (state.frameData.fileId === state.file?.id && ["hdf5", "parquet"].includes(state.frameData.mode)) loadFrameData(resolved, state.h5Compare.field);
    };
    $("#h5ComparePrev").addEventListener("click", () => change(state.h5Compare.index - 1)); $("#h5CompareNext").addEventListener("click", () => change(state.h5Compare.index + 1)); $("#h5CompareIndex").addEventListener("change", event => change(event.target.value)); $("#h5CompareSlider").addEventListener("input", event => change(event.target.value));
    loadH5Comparison(initial, state.h5Compare.field);
  }
  function renderPreviewTable(columns, rows) { const safeColumns = (columns || []).slice(0, 50); return `<div class="preview-table-wrap"><table class="preview-table"><thead><tr>${safeColumns.map(column => `<th>${esc(column)}</th>`).join("")}</tr></thead><tbody>${(rows || []).map(row => `<tr>${safeColumns.map(column => `<td title="${esc(previewValue(row?.[column]))}">${esc(previewValue(row?.[column]))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`; }
  function textPreviewSections(content) {
    const blocks = String(content || "").split(/\n\s*\n/).map(value => value.trim()).filter(Boolean).slice(0, 200);
    return blocks.map((value, index) => { const first = value.split("\n").find(line => line.trim()) || `段落 ${index + 1}`; const heading = first.match(/^#{1,6}\s+(.+)/)?.[1] || first; return { id: `text-${index}`, title: heading.slice(0, 52), content: value, kind: "paragraph" }; });
  }
  function unifiedPreviewShell(items, body, activeId = null) {
    return `<div class="unified-reviewer"><aside class="section-navigator"><div class="navigator-head"><strong>内容导航</strong><span>${items.length}</span></div><label class="navigator-search"><i data-lucide="search"></i><input id="previewNavSearch" placeholder="筛选段落或字段"></label><nav id="previewNavList">${items.map((item, index) => `<button class="navigator-item${item.id === activeId ? " active" : ""}" ${item.field ? `data-preview-field="${esc(item.id)}"` : `data-nav-target="${esc(item.id)}"`}><span>${esc(item.title)}</span><small>${esc(item.meta || `${index + 1}`)}</small></button>`).join("")}</nav></aside><article class="preview-document" id="previewDocument">${body}</article></div>`;
  }
  function bindUnifiedNavigation(content) {
    const search = $("#previewNavSearch", content), buttons = $$(".navigator-item", content);
    if (search) search.addEventListener("input", () => { const query = search.value.trim().toLowerCase(); buttons.forEach(button => button.classList.toggle("hidden", Boolean(query) && !button.textContent.toLowerCase().includes(query))); });
    $$('[data-nav-target]', content).forEach(button => button.addEventListener("click", () => { const target = document.getElementById(button.dataset.navTarget); if (!target) return; buttons.forEach(item => item.classList.toggle("active", item === button)); target.scrollIntoView({ behavior: "smooth", block: "start" }); }));
    $$('[data-preview-field]', content).forEach(button => button.addEventListener("click", () => openFilePreview(state.file, null, button.dataset.previewField)));
    const documentPane = $("#previewDocument", content), sections = $$(".document-section", content);
    if (documentPane && sections.length > 1) documentPane.addEventListener("scroll", () => { let current = sections[0]; for (const section of sections) if (section.offsetTop <= documentPane.scrollTop + 100) current = section; buttons.forEach(button => button.classList.toggle("active", button.dataset.navTarget === current.id)); }, { passive: true });
    lucide.createIcons();
  }
  function renderUnifiedPreview(preview) {
    $("#previewTitle").textContent = preview.name; $("#previewPath").textContent = preview.relative_path; $("#previewMode").textContent = String(preview.mode || "file").toUpperCase(); $("#previewControls").classList.add("hidden");
    const notice = $("#previewNotice"); notice.classList.toggle("hidden", !preview.truncated); notice.textContent = preview.truncated ? "为保证响应速度，当前显示安全截断预览。" : "";
    const content = $("#previewContent");
    if (preview.mode === "error") { content.innerHTML = `<div class="preview-error">${esc(preview.error || "无法读取文件")}</div>`; return; }
    if (preview.mode === "json") {
      const sections = preview.sections?.length ? preview.sections : [{ id: "json-0", title: "JSON", content: preview.content || "", kind: "document" }];
      const items = sections.map(item => ({ id: item.id, title: item.title, meta: item.kind }));
      const body = `<div class="document-header"><div><b>JSON 结构</b><span>${sections.length} 个段落</span></div><code>${fmtBytes(preview.size_bytes)}</code></div>${sections.map(item => `<section class="document-section" id="${esc(item.id)}"><header><h3>${esc(item.title)}</h3><span>${esc(item.kind || "value")}</span></header><pre>${esc(item.content || "")}</pre>${item.truncated ? '<small class="section-truncated">此段落已截断</small>' : ""}</section>`).join("")}`;
      content.innerHTML = unifiedPreviewShell(items, body, sections[0]?.id); bindUnifiedNavigation(content); return;
    }
    if (preview.mode === "text") {
      const sections = textPreviewSections(preview.content); const effective = sections.length ? sections : [{ id: "text-0", title: "空文件", content: "", kind: "paragraph" }];
      const items = effective.map(item => ({ id: item.id, title: item.title, meta: "段落" }));
      const body = `<div class="document-header"><div><b>文本内容</b><span>${effective.length} 个段落</span></div><code>${fmtBytes(preview.size_bytes)}</code></div>${effective.map((item, index) => `<section class="document-section text-section" id="${item.id}"><header><h3>${esc(item.title)}</h3><span>§ ${index + 1}</span></header><pre>${esc(item.content)}</pre></section>`).join("")}`;
      content.innerHTML = unifiedPreviewShell(items, body, effective[0]?.id); bindUnifiedNavigation(content); return;
    }
    if (["hdf5", "parquet"].includes(preview.mode)) {
      const parquet = preview.mode === "parquet";
      const datasets = preview.datasets || [], selected = datasets.find(item => item.path === preview.selected_field) || datasets[0] || {};
      const items = datasets.map(item => ({ id: item.path, title: item.path, meta: `${JSON.stringify(item.shape || [])} · ${item.dtype || ""}`, field: true }));
      const attributes = Object.entries(preview.attributes || {});
      const frameCount = Math.max(1, Number(selected.shape?.[0] || 1));
      const structureValue = parquet ? `${Number(preview.row_groups || 0)} Row Groups` : (selected.compression || "none");
      const structureLabel = parquet ? "存储结构" : "压缩";
      const body = `<section class="document-section dataset-section"><header><h3>${esc(preview.selected_field || "无数据字段")}</h3><span>${esc(selected.dtype || "")}</span></header><div class="dataset-facts"><div><b>${esc(JSON.stringify(selected.shape || []))}</b><span>完整 Shape</span></div><div><b>${frameCount}</b><span>可查阅帧数</span></div><div><b>${esc(structureValue)}</b><span>${structureLabel}</span></div></div>${attributes.length ? `<details><summary>文件属性 (${attributes.length})</summary><pre>${esc(JSON.stringify(Object.fromEntries(attributes), null, 2))}</pre></details>` : ""}<div class="h5-compare" id="h5Compare"><div class="h5-compare-toolbar"><strong>相邻帧对照</strong><button id="h5ComparePrev" type="button" title="上一组相邻帧">‹</button><label>起始帧 <input id="h5CompareIndex" type="number" min="0" value="0"></label><span id="h5CompareCount">/ ${frameCount - 1}</span><button id="h5CompareNext" type="button" title="下一组相邻帧">›</button><input id="h5CompareSlider" type="range" min="0" value="0"></div><div class="h5-compare-grid"><article><header><b id="h5CompareLeftTitle">第 0 帧</b><span id="h5CompareLeftMeta">读取中</span></header><pre id="h5CompareLeftValue"></pre></article><article><header><b id="h5CompareRightTitle">第 1 帧</b><span id="h5CompareRightMeta">读取中</span></header><pre id="h5CompareRightValue"></pre></article></div><section class="h5-diff" id="h5Diff"><header><div><strong>自动差异比较</strong><span id="h5DiffScope">等待真实帧值</span></div><b id="h5DiffStatus">比较中</b></header><div class="h5-diff-metrics"><div><span>变化元素</span><b id="h5DiffChanged">--</b></div><div><span>平均绝对差</span><b id="h5DiffMean">--</b></div><div><span>最大绝对差</span><b id="h5DiffMax">--</b></div></div><pre id="h5DiffPreview">正在比较真实帧值…</pre></section></div></section>`;
      content.innerHTML = unifiedPreviewShell(items, body, preview.selected_field); bindUnifiedNavigation(content); bindH5Comparison(selected); return;
    }
    if (preview.mode === "numpy") {
      const datasets = preview.datasets || [], selected = datasets.find(item => item.path === preview.selected_field) || datasets[0] || {};
      const items = datasets.map(item => ({ id: item.path, title: item.path, meta: `${JSON.stringify(item.shape || [])} · ${item.dtype || ""}`, field: true }));
      const body = `<div class="document-header"><div><b>NumPy 结构</b><span>${datasets.length} 个数据字段</span></div><code>${fmtBytes(preview.size_bytes)}</code></div><section class="document-section dataset-section"><header><h3>${esc(preview.selected_field || "无数据字段")}</h3><span>${esc(selected.dtype || "")}</span></header><div class="dataset-facts"><div><b>${esc(JSON.stringify(selected.shape || []))}</b><span>完整 Shape</span></div><div><b>${esc(JSON.stringify(preview.sample_shape || []))}</b><span>样本 Shape</span></div><div><b>n/a</b><span>压缩</span></div></div><div class="sample-heading">样本切片</div><pre>${esc(JSON.stringify(preview.sample, null, 2))}</pre></section>`;
      content.innerHTML = unifiedPreviewShell(items, body, preview.selected_field); bindUnifiedNavigation(content); return;
    }
    if (preview.mode === "table") { const columns = preview.columns || []; content.innerHTML = `<div class="document-header"><div><b>表格</b><span>${columns.length} 列${preview.row_count != null ? ` · ${preview.row_count} 行` : ""}</span></div><code>${fmtBytes(preview.size_bytes)}</code></div>${renderPreviewTable(columns, preview.rows)}`; return; }
    if (preview.mode === "binary") { content.innerHTML = `<pre class="preview-code">${esc(preview.hex || "")}</pre>`; return; }
    content.innerHTML = `<div class="preview-error">暂不支持该文件类型</div>`;
  }

  async function updateJointOverlayStatusLegacy() {
    const button = $("#jointOverlayButton");
    state.jointOverlayAvailable = false; state.jointOverlay = false; button.classList.remove("active"); button.disabled = true;
    if (!state.dataset || !state.episode) return;
    try {
      const status = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/joint-overlay/status`);
      state.jointOverlayAvailable = Boolean(status.available);
      button.disabled = false;
      button.title = state.jointOverlayAvailable ? `Joint 结构叠加 · ${status.joint_count || 0} 点` : (status.reason || "未找到 joint 数据");
    } catch (_) { button.disabled = false; button.title = "Joint 数据状态读取失败"; }
  }
  async function loadReviewAnnotations(episodeId, mediaFileId = null) {
    let stored = null;
    try { stored = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(episodeId)}/annotations`); } catch (_) {}
    const storedMatches = !stored?.source_video?.file_id || !mediaFileId || stored.source_video.file_id === mediaFileId;
    if (storedMatches && Number(stored?.summary?.manual_edit_count || 0) > 0) return stored;
    const [qwenTrim, yoloTrim] = await Promise.all([
      api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(episodeId)}/qwen-action-trim`).catch(() => null),
      api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(episodeId)}/no-action-trim`).catch(() => null),
    ]);
    const candidates = [storedMatches ? stored : null, qwenTrim, yoloTrim].filter(payload => payload && (!payload.source_video?.file_id || !mediaFileId || payload.source_video.file_id === mediaFileId));
    candidates.sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
    if (candidates.length) return candidates[0];
    throw new Error("当前视频流没有时序标注");
  }
  function isCurrentReviewSelection(datasetId, episodeId, mediaFileId, selectionToken = null) {
    return state.dataset?.id === datasetId
      && state.episode?.id === episodeId
      && (state.media?.file_id || null) === (mediaFileId || null)
      && (selectionToken == null || state.reviewSelectionToken === selectionToken);
  }
  async function loadCurationReport(episodeId, mediaFileId = null, selectionToken = null) {
    const datasetId = state.dataset?.id;
    if (!datasetId) return;
    const isCurrent = () => isCurrentReviewSelection(datasetId, episodeId, mediaFileId, selectionToken);
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(episodeId)}/curation`);
      if (!isCurrent()) return;
      if (payload.source_video?.file_id && mediaFileId && payload.source_video.file_id !== mediaFileId) { clearCurationReport(); return; }
      renderCurationReport(payload);
    } catch (_) { if (isCurrent()) clearCurationReport(); }
  }
  function clearActionMappingResult() { state.actionMapping = null; $("#actionMappingResult").classList.add("hidden"); $("#actionMappingWarning").classList.add("hidden"); $("#actionMappingSummary").textContent = ""; $("#actionMappingPath").textContent = ""; }
  function renderActionMappingResult(payload) {
    state.actionMapping = payload; const profile = payload.profile || {}, summary = payload.summary || {}, config = payload.config || {}, warning = (payload.warnings || []).join(" ");
    $("#actionMappingResult").classList.remove("hidden"); $("#actionMappingBadge").textContent = `${Number(summary.action_dim || profile.action_dim || 0)}D`;
    $("#actionMappingBadge").className = `badge ${summary.finite ? "ready" : ""}`;
    $("#actionMappingSummary").textContent = `${profile.name || config.profile_id || "Action"} · ${Number(summary.action_count || 0).toLocaleString()} 行 · ${config.coordinate_frame === "camera" ? "相机坐标" : "世界坐标"} · 未来 ${Number(config.horizon_frames || 0)} 帧${Number(profile.sides || 0) === 1 ? ` · ${config.source_hand === "left" ? "左手" : "右手"}` : ""}`;
    $("#actionMappingPath").textContent = payload.artifact_path || "";
    $("#actionMappingWarning").classList.toggle("hidden", !warning); $("#actionMappingWarning").textContent = warning;
  }
  async function loadActionMappingResult(episodeId, selectionToken = null) {
    const datasetId = state.dataset?.id; if (!datasetId) return;
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(episodeId)}/action-mapping`);
      if (state.dataset?.id !== datasetId || state.episode?.id !== episodeId || (selectionToken != null && state.reviewSelectionToken !== selectionToken)) return;
      renderActionMappingResult(payload);
    } catch (_) {
      if (state.dataset?.id === datasetId && state.episode?.id === episodeId && (selectionToken == null || state.reviewSelectionToken === selectionToken)) clearActionMappingResult();
    }
  }
  async function selectEpisode(id, mediaFileId = null) {
    const datasetId = state.dataset?.id, episode = state.dataset?.episodes.find(item => item.id === id);
    if (!datasetId || !episode) return;
    const selectionToken = ++state.reviewSelectionToken;
    stopPlayback(); resetNativePreview(); resetYoloOverlay(true); showMediaReview();
    state.episode = episode;
    state.media = (episode.media_streams || []).find(item => item.file_id === mediaFileId) || (episode.media_streams || []).find(item => item.file_id === episode.primary_media_file_id) || episode;
    const selectedMediaFileId = state.media?.file_id || null;
    state.frame = 0; state.annotations = null; state.behavior = null; state.curation = null; state.curationStageFilter = null; state.actionMapping = null;
    $("#statusEpisode").textContent = `EP: ${episode.name} / ${state.media.stream_name || state.media.relative_path || "primary"}`;
    $("#workspaceTitle").textContent = `Episode / ${episode.name}`;
    $("#workspaceMeta").textContent = `${state.media.stream_name || "primary"} · ${state.media.frame_count} frames · ${Number(state.media.fps).toFixed(2)} FPS`;
    showView("review"); enableEpisodeControls(); clearBehaviorAnnotation(); updateJointOverlayStatus(); clearCurationReport(); clearActionMappingResult();
    const annotationLoad = loadReviewAnnotations(id, selectedMediaFileId).then(payload => {
      if (!isCurrentReviewSelection(datasetId, id, selectedMediaFileId, selectionToken)) return;
      state.annotations = payload; renderAnnotations(payload);
    }).catch(() => {
      if (isCurrentReviewSelection(datasetId, id, selectedMediaFileId, selectionToken)) clearAnnotations();
    });
    await Promise.all([annotationLoad, loadCurationReport(id, selectedMediaFileId, selectionToken), loadActionMappingResult(id, selectionToken)]);
    if (!isCurrentReviewSelection(datasetId, id, selectedMediaFileId, selectionToken)) return;
    updateFrame(0); prepareNativePreview(id, selectedMediaFileId);
  }
  async function ensureEpisodeSelected() {
    if (state.episode) return true;
    const episode = state.dataset?.episodes?.[0];
    if (!episode) { toast("当前数据集没有可分析的 Episode", "error"); return false; }
    setStatus(`未手动选择 Episode，已自动选择 ${episode.name}`);
    await selectEpisode(episode.id, episode.primary_media_file_id || null);
    return state.episode?.id === episode.id;
  }
  function enableEpisodeControls() { const media = state.media || state.episode; ["#manualRangeButton", "#poseRecoveryButton", "#playButton", "#transportPlay", "#prevFrame", "#nextFrame", "#frameSlider", "#zoomIn", "#zoomOut"].forEach(id => $(id).disabled = false); $("#mediaInfo").textContent = `${media.width} x ${media.height} / ${Number(media.fps).toFixed(2)} FPS`; $("#totalTime").textContent = `/ ${fmtTime(media.duration)}`; $("#frameSlider").max = Math.max(0, media.frame_count - 1); $("#frameImage").classList.remove("hidden"); $("#videoPlayer").classList.add("hidden"); $("#viewerEmpty").classList.add("hidden"); updateYoloOverlayAvailability(); }

  function setPreviewStatus(message, mode = "running") { const node = $("#previewProxyStatus"); node.textContent = message; node.className = `badge native-preview-status ${mode}`; node.classList.remove("hidden"); }
  function resetJointIndices() {
    state.jointIndices = false; state.jointGeometryCurrent = null;
    const button = $("#jointIndexButton"); if (!button) return;
    button.disabled = true; button.classList.remove("active"); button.setAttribute("aria-pressed", "false");
  }
  function resetNativePreview() {
    state.previewPollToken += 1; state.nativePreview = null; state.nativePresentedFrame = -1; state.nativePresentedFrames = 0; state.nativeMediaTime = null; state.jointGeometryCurrent = null; state.jointGeometryFrame = -1; state.jointGeometryDesiredFrame = -1; state.jointGeometryInFlightFrame = -1; state.jointGeometryPendingFrame = null; state.jointGeometryRequestToken += 1; state.jointGeometryLastAt = 0; clearTimeout(state.jointGeometryTimer); state.jointGeometryTimer = null;
    if (state.jointGeometryAbortController) state.jointGeometryAbortController.abort(); state.jointGeometryAbortController = null;
    const video = $("#videoPlayer"); if (video) { video.pause(); video.removeAttribute("src"); video.load(); }
    $("#jointOverlayCanvas").classList.add("hidden"); $("#videoPlayer").classList.add("hidden"); $("#previewProxyStatus").classList.add("hidden");
  }
  async function prepareNativePreview(episodeId, mediaFileId, forceProxy = false) {
    const token = ++state.previewPollToken, query = `?media_file_id=${encodeURIComponent(mediaFileId || "")}${forceProxy ? "&force_proxy=true" : ""}`;
    setPreviewStatus(forceProxy ? "正在生成兼容预览" : "准备原生预览", "running");
    try {
      let payload = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(episodeId)}/preview-proxy${query}`, { method: "POST" });
      while (payload.status === "queued" || payload.status === "running") {
        if (token !== state.previewPollToken || state.episode?.id !== episodeId) return;
        setPreviewStatus(payload.message || `生成预览 ${Number(payload.progress || 0).toFixed(0)}%`, "running");
        await new Promise(resolve => setTimeout(resolve, 250));
        payload = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(episodeId)}/preview-proxy${query}`);
      }
      if (token !== state.previewPollToken || state.episode?.id !== episodeId) return;
      if (payload.status !== "ready") throw new Error(payload.error || payload.message || "预览代理生成失败");
      activateNativePreview(payload, episodeId, mediaFileId, forceProxy);
    } catch (error) {
      if (token !== state.previewPollToken || state.episode?.id !== episodeId) return;
      setPreviewStatus("原生预览不可用，使用逐帧模式", "failed");
      if (!forceProxy && String(error.message || "").includes("直接使用原始 MP4")) prepareNativePreview(episodeId, mediaFileId, true);
    }
  }
  function nativePreviewFps() { return Number(state.nativePreview?.fps || state.media?.fps || 30) || 30; }
  function nearestPtsIndex(points, seconds) {
    if (!Array.isArray(points) || !points.length || !Number.isFinite(seconds)) return null;
    let low = 0, high = points.length - 1;
    while (low < high) { const middle = Math.floor((low + high) / 2); if (Number(points[middle]) < seconds) low = middle + 1; else high = middle; }
    const right = low, left = Math.max(0, right - 1);
    return Math.abs(Number(points[left]) - seconds) <= Math.abs(Number(points[right]) - seconds) ? left : right;
  }
  function nativeFrameFromMediaTime(mediaTime) {
    const media = state.media || state.episode, preview = state.nativePreview;
    if (!media || !preview) return 0;
    const seconds = Number(mediaTime);
    let frame = Number.isFinite(seconds) ? Math.round(seconds * nativePreviewFps()) : state.frame;
    // Both source and FFmpeg passthrough proxies may expose an irregular PTS table.
    if (Array.isArray(preview.sourcePts) && preview.sourcePts.length) { const mapped = nearestPtsIndex(preview.sourcePts, seconds); if (mapped != null) frame = mapped; }
    const previewCount = Number(preview.frame_count || 0), mediaCount = Number(media.frame_count || 0), counts = [previewCount, mediaCount].filter(value => value > 0);
    const count = counts.length ? Math.max(1, Math.min(...counts)) : 1;
    return Math.max(0, Math.min(count - 1, frame));
  }
  function nativeMediaTimeForFrame(frame) {
    const preview = state.nativePreview, index = Math.max(0, Number(frame) || 0);
    if (Array.isArray(preview?.sourcePts) && Number.isFinite(Number(preview.sourcePts[index]))) return Number(preview.sourcePts[index]);
    return index / nativePreviewFps();
  }
  function revealNativePreview() { const video = $("#videoPlayer"), image = $("#frameImage"); image.classList.add("hidden"); video.classList.remove("hidden"); $("#viewerEmpty").classList.add("hidden"); }
  function activateNativePreview(payload, episodeId, mediaFileId, forceProxy) {
    const video = $("#videoPlayer");
    if (state.playing) stopPlayback();
    const preview = { ...payload, sourcePts: null, mappingPending: Boolean(payload.mapping_url), episodeId, mediaFileId };
    state.nativePreview = preview; state.nativePresentedFrame = -1; state.nativePresentedFrames = 0; state.nativeMediaTime = null;
    // Mapping is ancillary metadata; do not block the first decoded video frame on it.
    if (payload.mapping_url) api(payload.mapping_url).then(mapping => {
      if (state.nativePreview !== preview) return;
      preview.sourcePts = mapping.mapping?.source_pts_seconds || []; preview.mappingPending = false;
      if (video.readyState >= 2) updateNativeVideoFrame();
    }).catch(() => { if (state.nativePreview === preview) { preview.mappingPending = false; if (video.readyState >= 2) updateNativeVideoFrame(); } });
    const isCurrent = () => state.episode?.id === episodeId && state.nativePreview === preview;
    const syncLoadedFrame = () => { if (!isCurrent()) return; revealNativePreview(); updateNativeVideoFrame(); };
    video.onloadedmetadata = () => {
      if (!isCurrent()) return;
      const fps = nativePreviewFps(); video.currentTime = Math.max(0, nativeMediaTimeForFrame(state.frame));
      $("#mediaInfo").textContent = `${payload.width || state.media.width} x ${payload.height || state.media.height} / ${fps.toFixed(2)} FPS`;
      setPreviewStatus(payload.delivery === "source" ? "原始视频" : `原生代理 · ${String(payload.codec || "VP8").toUpperCase()}`, "ready");
      if (video.readyState >= 2) syncLoadedFrame();
    };
    video.onloadeddata = syncLoadedFrame;
    video.onerror = () => { if (!isCurrent()) return; if (payload.delivery === "source" && !forceProxy) prepareNativePreview(episodeId, mediaFileId, true); else setPreviewStatus("原生预览失败，使用逐帧模式", "failed"); };
    video.onended = () => { if (!isCurrent()) return; const media = state.media || state.episode; if (media?.frame_count) updateNativeVideoFrame(); stopPlayback(); };
    video.muted = true; video.playsInline = true; video.src = `${payload.media_url}&v=${Date.now()}`; video.load();
  }
  function updateFrameDisplay(frame) {
    const media = state.media || state.episode; if (!media?.frame_count) return;
    state.frame = Math.max(0, Math.min(Number(frame) || 0, media.frame_count - 1)); $("#frameSlider").value = state.frame;
    const sourcePts = state.nativePreview?.sourcePts, displayFps = state.nativePreview ? nativePreviewFps() : Number(media.fps || 30), time = sourcePts?.[state.frame] ?? (state.frame / displayFps);
    $("#currentTime").textContent = fmtTime(time); $("#frameNumber").textContent = `FRAME ${state.frame + 1} / ${media.frame_count}`; $("#frameNumber").classList.remove("hidden"); updateTimelineCursor(); renderCurrentFrame(); scheduleFrameData(state.frame); renderYoloOverlayFrame(state.frame);
  }
  function scheduleNativeFrameCallback() {
    const video = $("#videoPlayer"), playbackToken = state.playbackToken, preview = state.nativePreview;
    if (!state.playing || !preview || video.paused) return;
    if (video.requestVideoFrameCallback) {
      state.nativeFrameCallback = video.requestVideoFrameCallback((_, metadata) => {
        state.nativeFrameCallback = null;
        if (!state.playing || playbackToken !== state.playbackToken || state.nativePreview !== preview || video.paused) return;
        updateNativeVideoFrame(metadata); scheduleNativeFrameCallback();
      });
    } else {
      state.timer = setTimeout(() => {
        state.timer = null;
        if (!state.playing || playbackToken !== state.playbackToken || state.nativePreview !== preview || video.paused) return;
        updateNativeVideoFrame(); scheduleNativeFrameCallback();
      }, Math.max(15, Math.round(1000 / nativePreviewFps())));
    }
  }
  function updateNativeVideoFrame(metadata = null) {
    const media = state.media || state.episode, video = $("#videoPlayer"); if (!media || !state.nativePreview) return;
    const mediaTime = Number.isFinite(Number(metadata?.mediaTime)) ? Number(metadata.mediaTime) : Number(video.currentTime);
    const frame = nativeFrameFromMediaTime(mediaTime), previousFrame = state.nativePresentedFrame;
    state.nativePresentedFrame = frame; state.nativeMediaTime = Number.isFinite(mediaTime) ? mediaTime : null;
    if (Number.isFinite(Number(metadata?.presentedFrames))) state.nativePresentedFrames = Number(metadata.presentedFrames);
    video.dataset.presentedFrame = String(frame);
    if (Number.isFinite(mediaTime)) video.dataset.mediaTime = String(mediaTime);
    if (state.nativePresentedFrames) video.dataset.presentedFrames = String(state.nativePresentedFrames);
    if (frame !== previousFrame || frame !== state.frame) updateFrameDisplay(frame);
    requestJointGeometry(frame);
  }
  function clearYoloOverlayCanvas() {
    const canvas = $("#yoloOverlayCanvas"), ctx = canvas?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas?.classList.add("hidden");
    if (canvas) { delete canvas.dataset.yoloFrame; delete canvas.dataset.yoloCurrentFrame; delete canvas.dataset.yoloExact; }
    state.yoloOverlaySampleFrame = -1;
  }
  function resetYoloOverlay(clearToggle = false) {
    state.yoloOverlayLoadToken += 1; state.yoloOverlayReport = null; state.yoloOverlaySamples = []; clearYoloOverlayCanvas();
    if (!clearToggle) return;
    state.yoloOverlay = false;
    const button = $("#yoloOverlayButton"), hint = $("#yoloOverlayHint");
    button?.classList.remove("active"); button?.setAttribute("aria-pressed", "false");
    hint?.classList.add("hidden"); if (hint) { hint.textContent = ""; hint.title = ""; }
  }
  function updateYoloOverlayAvailability() {
    const button = $("#yoloOverlayButton"), local = state.models?.local || {};
    if (!button) return;
    const available = Boolean(state.episode && local.loaded && local.family === "YOLOE");
    button.disabled = !available;
    button.title = available ? "显示当前视频实际用于无动作剪切的 YOLOE 采样框" : (state.episode ? "需要已加载的 YOLOE 分割模型" : "请先选择 Episode");
    if (!available && state.yoloOverlay) resetYoloOverlay(true);
  }
  function installYoloOverlayReport(payload) {
    const sourceFileId = payload?.source_video?.file_id || null, mediaFileId = state.media?.file_id || null;
    if (sourceFileId && mediaFileId && sourceFileId !== mediaFileId) throw new Error("当前视频尚未运行 YOLOE 无动作剪切");
    const samples = (payload?.samples || []).filter(item => Number.isFinite(Number(item.frame)) && Array.isArray(item.detections)).sort((a, b) => Number(a.frame) - Number(b.frame));
    if (!samples.length) throw new Error("当前 YOLOE 剪切结果没有可显示的采样检测框");
    state.yoloOverlayReport = payload; state.yoloOverlaySamples = samples;
    const prompts = payload.prompt_classes || payload.primary_terms || [], button = $("#yoloOverlayButton");
    button.title = `YOLOE 采样框 · ${samples.length} 帧${prompts.length ? ` · ${prompts.join(", ")}` : ""}`;
    renderYoloOverlayFrame(state.frame);
  }
  async function loadYoloOverlayReport() {
    const datasetId = state.dataset?.id, episodeId = state.episode?.id, mediaFileId = state.media?.file_id || null, token = ++state.yoloOverlayLoadToken, hint = $("#yoloOverlayHint");
    if (!datasetId || !episodeId || !state.yoloOverlay) return;
    hint.classList.remove("hidden"); hint.textContent = "YOLOE · 载入采样框"; hint.title = "读取 .alicePD 中的真实剪切检测结果";
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(episodeId)}/no-action-trim`);
      const current = token === state.yoloOverlayLoadToken && state.yoloOverlay && state.dataset?.id === datasetId && state.episode?.id === episodeId && (state.media?.file_id || null) === mediaFileId;
      if (!current) return;
      installYoloOverlayReport(payload);
    } catch (error) {
      const current = token === state.yoloOverlayLoadToken && state.yoloOverlay && state.dataset?.id === datasetId && state.episode?.id === episodeId && (state.media?.file_id || null) === mediaFileId;
      if (!current) return;
      resetYoloOverlay(true); hint.classList.remove("hidden"); hint.textContent = "YOLOE · 无可用框"; hint.title = error.message; toast(error.message, "error"); setStatus(error.message);
    }
  }
  function nearestYoloOverlaySample(frame) {
    const samples = state.yoloOverlaySamples; if (!samples.length) return null;
    let low = 0, high = samples.length - 1;
    while (low < high) { const middle = Math.floor((low + high) / 2); if (Number(samples[middle].frame) < frame) low = middle + 1; else high = middle; }
    const right = low, left = Math.max(0, right - 1);
    return Math.abs(Number(samples[left].frame) - frame) <= Math.abs(Number(samples[right].frame) - frame) ? samples[left] : samples[right];
  }
  function renderYoloOverlayFrame(frame) {
    const canvas = $("#yoloOverlayCanvas"), viewer = $("#viewer"), image = $("#frameImage"), video = $("#videoPlayer"), hint = $("#yoloOverlayHint");
    if (!state.yoloOverlay || !state.yoloOverlayReport || !state.yoloOverlaySamples.length) { clearYoloOverlayCanvas(); return; }
    const requestedFrame = Math.max(0, Math.round(Number(frame) || 0));
    const nativeVisible = state.nativePreview && !video.classList.contains("hidden"), displayedFrame = nativeVisible ? state.nativePresentedFrame : Number(image.dataset.presentedFrame);
    if (!Number.isFinite(displayedFrame) || displayedFrame !== requestedFrame || state.frame !== requestedFrame) { clearYoloOverlayCanvas(); return; }
    const sample = nearestYoloOverlaySample(requestedFrame); if (!sample) { clearYoloOverlayCanvas(); return; }
    const media = state.media || state.episode, sourceWidth = Number(state.yoloOverlayReport.source_video?.width || media?.width || image.naturalWidth || video.videoWidth || 0), sourceHeight = Number(state.yoloOverlayReport.source_video?.height || media?.height || image.naturalHeight || video.videoHeight || 0);
    const width = viewer.clientWidth, height = viewer.clientHeight;
    if (!(sourceWidth > 0 && sourceHeight > 0 && width > 0 && height > 0)) { clearYoloOverlayCanvas(); return; }
    const dpr = window.devicePixelRatio || 1; canvas.width = Math.max(1, Math.round(width * dpr)); canvas.height = Math.max(1, Math.round(height * dpr)); canvas.classList.remove("hidden");
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width / sourceWidth, height / sourceHeight), offsetX = (width - sourceWidth * scale) / 2, offsetY = (height - sourceHeight * scale) / 2, exact = Number(sample.frame) === requestedFrame;
    const detections = sample.detections || []; ctx.lineWidth = 2; ctx.font = '11px "Segoe UI", sans-serif'; ctx.textBaseline = "top";
    for (const detection of detections) {
      const box = (detection.box || []).map(Number); if (box.length !== 4 || box.some(value => !Number.isFinite(value))) continue;
      const label = String(detection.label || "object"), lowered = label.toLowerCase(), robot = lowered.includes("robot") || lowered.includes("gripper"), hand = detection.group === "hand" || lowered.includes("hand") || robot;
      const color = robot ? "#58a6ff" : hand ? "#14d8e8" : "#ffd24a", x = offsetX + box[0] * scale, y = offsetY + box[1] * scale, boxWidth = Math.max(1, (box[2] - box[0]) * scale), boxHeight = Math.max(1, (box[3] - box[1]) * scale);
      ctx.strokeStyle = color; ctx.setLineDash(exact ? [] : [6, 4]); ctx.strokeRect(x, y, boxWidth, boxHeight); ctx.setLineDash([]);
      const confidence = Number(detection.confidence || 0), text = `${label} ${Math.round(confidence * 100)}%`, textWidth = Math.min(boxWidth, ctx.measureText(text).width + 10), labelY = y >= 19 ? y - 18 : y + 2;
      ctx.fillStyle = "rgba(9,17,22,.88)"; ctx.fillRect(x, labelY, Math.max(34, textWidth), 17); ctx.fillStyle = color; ctx.fillText(text, x + 5, labelY + 2, Math.max(24, boxWidth - 10));
    }
    const handCount = detections.filter(item => item.group === "hand").length, objectCount = detections.length - handCount, sampleFrame = Number(sample.frame); state.yoloOverlaySampleFrame = sampleFrame;
    canvas.dataset.yoloFrame = String(sampleFrame); canvas.dataset.yoloCurrentFrame = String(requestedFrame); canvas.dataset.yoloExact = String(exact);
    hint.classList.remove("hidden"); hint.textContent = `YOLOE · ${handCount} 手 · ${objectCount} 目标 · F${sampleFrame + 1}${exact ? "" : ` → F${requestedFrame + 1}`}`;
    hint.title = exact ? `当前帧的真实 YOLOE 采样框` : `虚线框来自最近采样帧 F${sampleFrame + 1}，当前显示 F${requestedFrame + 1}`;
  }
  function toggleYoloOverlay() {
    const button = $("#yoloOverlayButton"), hint = $("#yoloOverlayHint"); if (button.disabled) return;
    state.yoloOverlay = !state.yoloOverlay; state.yoloOverlayLoadToken += 1; button.classList.toggle("active", state.yoloOverlay); button.setAttribute("aria-pressed", String(state.yoloOverlay));
    if (!state.yoloOverlay) { clearYoloOverlayCanvas(); hint.classList.add("hidden"); hint.textContent = ""; return; }
    if (state.yoloOverlayReport) renderYoloOverlayFrame(state.frame); else loadYoloOverlayReport();
  }
  function placeJointIndexBadge(pointX, pointY, badgeWidth, badgeHeight, width, height, occupied) {
    const directions = [[.7, -.7], [-.7, -.7], [1, 0], [-1, 0], [0, -1], [.7, .7], [-.7, .7], [0, 1]];
    let fallback = { x: pointX + 6, y: pointY - badgeHeight - 3, radius: 11 };
    for (const radius of [11, 20, 29, 38, 47, 56, 65]) {
      for (const [directionX, directionY] of directions) {
        const x = Math.max(1, Math.min(width - badgeWidth - 1, Math.round(pointX + directionX * radius - badgeWidth / 2)));
        const y = Math.max(1, Math.min(height - badgeHeight - 1, Math.round(pointY + directionY * radius - badgeHeight / 2)));
        const overlaps = occupied.some(box => x < box.right + 2 && x + badgeWidth + 2 > box.left && y < box.bottom + 2 && y + badgeHeight + 2 > box.top);
        fallback = { x, y, radius }; if (!overlaps) { occupied.push({ left: x, top: y, right: x + badgeWidth, bottom: y + badgeHeight }); return fallback; }
      }
    }
    occupied.push({ left: fallback.x, top: fallback.y, right: fallback.x + badgeWidth, bottom: fallback.y + badgeHeight }); return fallback;
  }
  function drawJointGeometry(geometry) {
    const canvas = $("#jointOverlayCanvas"), viewer = $("#viewer"); if (!geometry || !state.jointOverlay || !state.nativePreview) { canvas.classList.add("hidden"); return; }
    const geometryFrame = Number(geometry.frame_index ?? state.jointGeometryFrame);
    if (!Number.isFinite(geometryFrame) || geometryFrame !== state.frame || geometryFrame !== state.nativePresentedFrame || geometryFrame !== state.jointGeometryDesiredFrame) { canvas.classList.add("hidden"); return; }
    const dpr = window.devicePixelRatio || 1, width = viewer.clientWidth, height = viewer.clientHeight; canvas.width = Math.max(1, Math.round(width * dpr)); canvas.height = Math.max(1, Math.round(height * dpr)); canvas.classList.remove("hidden");
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    const scale = Math.min(width / geometry.width, height / geometry.height), offsetX = (width - geometry.width * scale) / 2, offsetY = (height - geometry.height * scale) / 2, points = geometry.points || [];
    const color = side => side === "left" ? "#4080eb" : side === "right" ? "#ebb448" : "#d9e0e5", pointRadius = 2.5;
    ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const [start, end] of geometry.edges || []) { const a = points[start], b = points[end]; if (!a || !b) continue; ctx.strokeStyle = color(a.side); ctx.beginPath(); ctx.moveTo(offsetX + a.x * scale, offsetY + a.y * scale); ctx.lineTo(offsetX + b.x * scale, offsetY + b.y * scale); ctx.stroke(); }
    for (const point of points) { ctx.fillStyle = color(point.side); ctx.beginPath(); ctx.arc(offsetX + point.x * scale, offsetY + point.y * scale, pointRadius, 0, Math.PI * 2); ctx.fill(); }
    if (state.jointIndices) {
      ctx.font = '600 9px "Segoe UI", sans-serif'; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const occupied = [];
      for (const point of points) {
        const sourceIndex = Number(point.source_index); if (!Number.isInteger(sourceIndex) || sourceIndex < 0) continue;
        const label = String(sourceIndex), pointX = offsetX + point.x * scale, pointY = offsetY + point.y * scale;
        const badgeWidth = Math.max(13, Math.ceil(ctx.measureText(label).width) + 6), badgeHeight = 13;
        const placement = placeJointIndexBadge(pointX, pointY, badgeWidth, badgeHeight, width, height, occupied), badgeX = placement.x, badgeY = placement.y;
        if (placement.radius > 16) { ctx.strokeStyle = "rgba(235,240,244,.55)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pointX, pointY); ctx.lineTo(badgeX + badgeWidth / 2, badgeY + badgeHeight / 2); ctx.stroke(); }
        ctx.fillStyle = "rgba(15,20,24,.88)"; ctx.beginPath();
        if (typeof ctx.roundRect === "function") ctx.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, 3); else ctx.rect(badgeX, badgeY, badgeWidth, badgeHeight);
        ctx.fill(); ctx.fillStyle = "#fff"; ctx.fillText(label, badgeX + badgeWidth / 2, badgeY + badgeHeight / 2 + .5);
      }
    }
    canvas.dataset.jointFrame = String(geometry.frame_index ?? state.jointGeometryFrame);
    const hint = $("#jointOverlayHint"), multiplier = Number(geometry.alignment_multiplier || 1), hz = Number(geometry.sensor_hz || 0);
    hint.classList.remove("hidden");
    hint.textContent = geometry.alignment_valid === false ? "SYNC 当前帧缺少传感器样本" : `SYNC ${hz ? `${hz.toFixed(3)} Hz · ` : ""}×${multiplier.toFixed(4)} · row ${geometry.sensor_index ?? "--"}`;
    hint.title = `视频帧 ${geometry.frame_index ?? state.frame} → 传感器行 ${geometry.sensor_index ?? "缺测"} · ${geometry.alignment_mode || "identity"}`;
  }
  function hideStaleJointGeometry(frame) {
    const canvas = $("#jointOverlayCanvas"), shownFrame = Number(canvas.dataset.jointFrame);
    if (Number.isFinite(shownFrame) && shownFrame === frame) return;
    canvas.classList.add("hidden"); delete canvas.dataset.jointFrame;
  }
  function invalidateJointGeometry(frame, abort = false) {
    const requestedFrame = Math.max(0, Math.round(Number(frame) || 0));
    state.jointGeometryDesiredFrame = requestedFrame; state.jointGeometryPendingFrame = null; state.jointGeometryRequestToken += 1;
    hideStaleJointGeometry(requestedFrame);
    if (!abort) return;
    clearTimeout(state.jointGeometryTimer); state.jointGeometryTimer = null;
    if (state.jointGeometryAbortController) state.jointGeometryAbortController.abort();
  }
  function requestJointGeometry(frame, force = false) {
    if (!state.nativePreview || !state.jointOverlay || !state.dataset || !state.episode || state.nativePreview.mappingPending) { state.jointGeometryPendingFrame = null; $("#jointOverlayCanvas").classList.add("hidden"); return; }
    if (state.nativePreview.mappingPending) { state.jointGeometryPendingFrame = null; hideStaleJointGeometry(Math.max(0, Math.round(Number(frame) || 0))); return; }
    const requestedFrame = Math.max(0, Math.round(Number(frame) || 0)), changed = requestedFrame !== state.jointGeometryDesiredFrame;
    if (!force && !changed && (state.jointGeometryFrame === requestedFrame || state.jointGeometryInFlightFrame === requestedFrame || state.jointGeometryPendingFrame === requestedFrame)) return;
    if (changed || force) state.jointGeometryRequestToken += 1;
    state.jointGeometryDesiredFrame = requestedFrame; state.jointGeometryPendingFrame = requestedFrame; hideStaleJointGeometry(requestedFrame);
    if (force) state.jointGeometryFrame = -1;
    if (force && state.jointGeometryAbortController) { state.jointGeometryAbortController.abort(); return; }
    if (state.jointGeometryAbortController || state.jointGeometryTimer) return;
    const delay = force ? 0 : Math.max(0, 30 - (Date.now() - state.jointGeometryLastAt));
    state.jointGeometryTimer = setTimeout(drainJointGeometry, delay);
  }
  async function drainJointGeometry() {
    state.jointGeometryTimer = null;
    if (!state.nativePreview || !state.jointOverlay || !state.dataset || !state.episode || state.jointGeometryAbortController) return;
    const frame = state.jointGeometryPendingFrame ?? state.frame; state.jointGeometryPendingFrame = null; state.jointGeometryLastAt = Date.now();
    const datasetId = state.dataset.id, episodeId = state.episode.id, mediaFileId = state.media?.file_id || "", preview = state.nativePreview, requestToken = state.jointGeometryRequestToken, controller = new AbortController(); state.jointGeometryAbortController = controller; state.jointGeometryInFlightFrame = frame;
    try {
      const query = `?index=${frame}&media_file_id=${encodeURIComponent(mediaFileId)}`;
      const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(episodeId)}/joint-overlay/frame${query}`, { cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error("Joint geometry unavailable");
      const geometry = await response.json();
      const geometryFrame = Number(geometry.frame_index ?? frame);
      const current = state.jointOverlay && state.nativePreview === preview && state.dataset?.id === datasetId && state.episode?.id === episodeId && state.jointGeometryRequestToken === requestToken && state.jointGeometryDesiredFrame === frame && state.nativePresentedFrame === frame && state.frame === frame && geometryFrame === frame;
      if (current) { state.jointGeometryFrame = frame; state.jointGeometryCurrent = geometry; drawJointGeometry(geometry); }
    } catch (error) { if (error.name !== "AbortError" && state.jointGeometryFrame < 0) $("#jointOverlayCanvas").classList.add("hidden"); }
    finally {
      if (state.jointGeometryAbortController === controller) state.jointGeometryAbortController = null;
      if (state.jointGeometryInFlightFrame === frame) state.jointGeometryInFlightFrame = -1;
      const nextFrame = state.jointGeometryPendingFrame;
      if (nextFrame != null && state.jointOverlay && state.nativePreview) { state.jointGeometryPendingFrame = null; requestJointGeometry(nextFrame); }
    }
  }
  async function updateJointOverlayStatus() {
    const button = $("#jointOverlayButton"), hint = $("#jointOverlayHint"), episodeId = state.episode?.id;
    $("#behaviorAnnotateButton").disabled = !state.dataset?.episodes?.length;
    state.jointOverlayAvailable = false; state.jointOverlay = false; resetJointIndices(); button.classList.remove("active"); button.setAttribute("aria-pressed", "false"); button.disabled = true; hint.classList.add("hidden"); hint.textContent = "";
    if (!state.dataset || !episodeId) return;
    loadBehaviorAnnotation();
    try {
      const status = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(episodeId)}/joint-overlay/status`);
      if (state.episode?.id !== episodeId) return;
      state.jointOverlayAvailable = Boolean(status.available);
      button.disabled = !state.jointOverlayAvailable;
      button.title = state.jointOverlayAvailable ? `Joint 结构叠加 · ${status.joint_count || 0} 点` : (status.reason || "未找到可投影的 joint 数据");
      if (!state.jointOverlayAvailable) { hint.classList.remove("hidden"); hint.textContent = status.initial_position_available === false && status.reason ? status.reason : (status.joint_state_available ? "仅有关节角，缺少投影标定" : "没有可投影骨架"); hint.title = button.title; }
    } catch (_) {
      if (state.episode?.id !== episodeId) return;
      button.disabled = true; button.title = "Joint 数据状态读取失败"; hint.classList.remove("hidden"); hint.textContent = "Joint 状态读取失败";
    }
  }
  function updateFrame(frame, playbackRequest = false) {
    if (!state.episode) return Promise.resolve(false);
    if (!playbackRequest && state.playing) stopPlayback();
    const media = state.media || state.episode;
    state.frame = Math.max(0, Math.min(Number(frame) || 0, media.frame_count - 1)); updateFrameDisplay(state.frame);
    if (state.nativePreview?.ready || state.nativePreview?.status === "ready") { const video = $("#videoPlayer"), targetTime = nativeMediaTimeForFrame(state.frame); if (Math.abs((video.currentTime || 0) - targetTime) > 0.001) { invalidateJointGeometry(state.frame, true); video.currentTime = targetTime; } else updateNativeVideoFrame(); return Promise.resolve(true); }
    const overlay = $(".segmented button.active")?.dataset.display !== "source", mediaQuery = media.file_id ? `&media_file_id=${encodeURIComponent(media.file_id)}` : "", image = $("#frameImage");
    if (state.frameAbortController) state.frameAbortController.abort();
    if (state.frameImageResolve) state.frameImageResolve(false);
    const requestToken = ++state.frameImageToken;
    const requestedImageFrame = state.frame, controller = new AbortController(), sourceUrl = `/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/frame?index=${requestedImageFrame}&overlay=${overlay}&joint_overlay=${state.jointOverlay}&joint_indices=${state.jointIndices}${mediaQuery}&t=${Date.now()}`;
    state.frameAbortController = controller;
    const loaded = new Promise(resolve => {
      let settled = false, pendingObjectUrl = null;
      const finish = success => {
        if (settled) return;
        settled = true; clearTimeout(state.frameImageTimeout); if (!success && pendingObjectUrl && state.frameObjectUrl !== pendingObjectUrl) URL.revokeObjectURL(pendingObjectUrl);
        if (requestToken === state.frameImageToken) { image.onload = null; image.onerror = null; state.frameImageResolve = null; state.frameImageTimeout = null; state.frameAbortController = null; }
        resolve(Boolean(success));
      };
      state.frameImageResolve = finish; state.frameImageTimeout = setTimeout(() => { controller.abort(); finish(false); }, 5000);
      fetch(sourceUrl, { cache: "no-store", signal: controller.signal }).then(response => {
        if (!response.ok) throw new Error(`帧读取失败 (${response.status})`);
        return response.blob();
      }).then(blob => {
        if (requestToken !== state.frameImageToken) { finish(false); return; }
        const objectUrl = URL.createObjectURL(blob), previousUrl = state.frameObjectUrl; pendingObjectUrl = objectUrl;
        image.onload = () => { if (requestToken !== state.frameImageToken || state.frame !== requestedImageFrame) { URL.revokeObjectURL(objectUrl); finish(false); return; } if (previousUrl) URL.revokeObjectURL(previousUrl); state.frameObjectUrl = objectUrl; image.dataset.presentedFrame = String(requestedImageFrame); renderYoloOverlayFrame(requestedImageFrame); finish(true); };
        image.onerror = () => { URL.revokeObjectURL(objectUrl); finish(false); };
        image.src = objectUrl;
      }).catch(error => { if (error.name !== "AbortError") console.warn(error.message); finish(false); });
    });
    return loaded;
  }
  function timelineIntervalItems(segments, declaredFrameCount = 0) {
    const normalized = (Array.isArray(segments) ? segments : []).map(segment => {
      const startFrame = Math.max(0, Math.round(Number(segment.start_frame || 0)));
      const endFrame = Math.max(startFrame, Math.round(Number(segment.end_frame ?? startFrame)));
      return { ...segment, start_frame: startFrame, end_frame: endFrame };
    }).sort((a, b) => a.start_frame - b.start_frame);
    if (!normalized.length) return [];
    const frameCount = Math.max(1, Number(declaredFrameCount || 0), ...normalized.map(segment => segment.end_frame + 1));
    const items = []; let cursor = 0;
    for (const segment of normalized) {
      if (segment.end_frame < cursor) continue;
      const startFrame = Math.max(cursor, Math.min(frameCount - 1, segment.start_frame));
      const endFrame = Math.max(startFrame, Math.min(frameCount - 1, segment.end_frame));
      if (startFrame > cursor) items.push({ gap: true, start_frame: cursor, end_frame: startFrame - 1 });
      if (endFrame >= cursor) items.push({ ...segment, start_frame: startFrame, end_frame: endFrame });
      cursor = Math.max(cursor, endFrame + 1);
      if (cursor >= frameCount) break;
    }
    if (cursor < frameCount) items.push({ gap: true, start_frame: cursor, end_frame: frameCount - 1 });
    return items;
  }
  function renderActionTrack() {
    const track = $("#segmentTrack"), frameCount = Number(state.media?.frame_count || state.episode?.frame_count || state.behavior?.source_video?.frame_count || 0);
    const behaviorSegments = Array.isArray(state.behavior?.segments) ? state.behavior.segments : [];
    if (behaviorSegments.length) {
      const items = timelineIntervalItems(behaviorSegments, frameCount); let phaseIndex = 0;
      track.innerHTML = items.map(segment => {
        const duration = segment.end_frame - segment.start_frame + 1;
        if (segment.gap) return `<span class="segment behavior-phase-gap" aria-hidden="true" style="flex:${duration}"></span>`;
        const phase = readableBehaviorPhase(segment), description = behaviorReadableLabel(segment.description) || behaviorReadableLabel(segment.label) || phase;
        const boundary = behaviorBoundarySource(segment, state.behavior), sourceLabel = boundary ? ` · ${boundary.label}` : "";
        const title = `${phase} · F${segment.start_frame + 1}–F${segment.end_frame + 1} · ${description}${sourceLabel}`;
        const tone = behaviorPhaseTone(segment, phaseIndex++);
        return `<button type="button" class="segment behavior-phase-segment phase-tone-${tone}" data-frame="${segment.start_frame}" title="${escAttr(title)}" aria-label="${escAttr(title)}" style="flex:${duration}"><span>${esc(phase)}</span></button>`;
      }).join("") + '<i class="timeline-cursor" aria-hidden="true"></i>';
      $$("button.behavior-phase-segment", track).forEach(button => button.addEventListener("click", () => updateFrame(Number(button.dataset.frame))));
      return;
    }
    const actionSegments = Array.isArray(state.annotations?.segments) ? state.annotations.segments : [];
    track.innerHTML = actionSegments.map(item => `<button type="button" class="segment annotation-state-segment ${item.state === "valid" ? "good" : item.state === "invalid" ? "bad" : "uncertain"}" data-frame="${Number(item.start_frame || 0)}" title="${escAttr(item.reason || item.state || "")}" style="flex:${Math.max(1, Number(item.end_frame || 0) - Number(item.start_frame || 0) + 1)}"></button>`).join("") + (actionSegments.length ? '<i class="timeline-cursor" aria-hidden="true"></i>' : "");
    $$("button.annotation-state-segment", track).forEach(button => button.addEventListener("click", () => updateFrame(Number(button.dataset.frame))));
  }
  function renderTrimTrack() {
    const group = $("#trimTrackGroup"), track = $("#trimTrack");
    const behaviorSegments = Array.isArray(state.behavior?.segments) ? state.behavior.segments : [];
    const trimSegments = Array.isArray(state.annotations?.segments) ? state.annotations.segments : [];
    const visible = Boolean(behaviorSegments.length && trimSegments.length); group.classList.toggle("hidden", !visible);
    if (!visible) { track.innerHTML = ""; return; }
    const frameCount = Number(state.media?.frame_count || state.episode?.frame_count || 0), items = timelineIntervalItems(trimSegments, frameCount);
    track.innerHTML = items.map(item => {
      const duration = item.end_frame - item.start_frame + 1;
      if (item.gap) return `<span class="segment trim-state-gap" aria-hidden="true" style="flex:${duration}"></span>`;
      const stateClass = item.state === "valid" ? "good" : item.state === "invalid" ? "bad" : "uncertain";
      const stateLabel = item.state === "valid" ? "有效" : item.state === "invalid" ? "无效" : "待确认";
      return `<button type="button" class="segment trim-state-segment ${stateClass}" data-frame="${item.start_frame}" title="${escAttr(`${stateLabel} · ${item.reason || "剪切片段"}`)}" style="flex:${duration}"></button>`;
    }).join("") + '<i class="timeline-cursor" aria-hidden="true"></i>';
    $$("button.trim-state-segment", track).forEach(button => button.addEventListener("click", () => updateFrame(Number(button.dataset.frame))));
  }
  function refreshTimelineVisibility() {
    renderActionTrack(); renderTrimTrack();
    const behaviorSegments = Array.isArray(state.behavior?.segments) ? state.behavior.segments : [], actionSegments = Array.isArray(state.annotations?.segments) ? state.annotations.segments : [], hasCuration = Boolean(state.curation?.segments?.length), visible = Boolean(behaviorSegments.length || actionSegments.length || hasCuration);
    $("#timelineEmpty").classList.toggle("hidden", visible); $("#timeline").classList.toggle("hidden", !visible); $("#curationTrackGroup").classList.toggle("hidden", !hasCuration);
    const actionLabel = behaviorSegments.length ? `${behaviorSegments.length} 个动作阶段` : actionSegments.length ? `${actionSegments.length} 个动作片段` : "无动作标注", trimLabel = behaviorSegments.length && actionSegments.length ? ` · ${actionSegments.length} 个剪切片段` : "", curationLabel = hasCuration ? `${Number(state.curation.summary?.invalid_frame_count || 0).toLocaleString()} 个质量异常帧` : "无质量报告";
    $("#clipSummary").textContent = visible ? `${actionLabel}${trimLabel} · ${curationLabel}` : "尚未分析";
    updateTimelineCursor();
  }
  function clearAnnotations() { state.annotations = null; $("#motionSeries").innerHTML = ""; $("#clipEmpty").classList.remove("hidden"); $("#clipTable").classList.add("hidden"); $("#resultEmpty").classList.remove("hidden"); $("#resultBox").classList.add("hidden"); updateBehaviorRemovalState(); refreshTimelineVisibility(); renderCurrentFrame(); }
  function updateTimelineCursor() { const media = state.media || state.episode; if (!media?.frame_count) return; $$(".timeline-cursor", $("#timeline")).forEach(cursor => { cursor.style.left = `${Math.max(0, Math.min(100, state.frame / Math.max(1, media.frame_count - 1) * 100))}%`; cursor.title = `Frame ${state.frame + 1}`; }); }
  function renderAnnotations(payload) { const segments = payload?.segments || [], samples = payload?.samples || []; state.annotations = payload; $("#clipEmpty").classList.toggle("hidden", Boolean(segments.length)); $("#clipTable").classList.toggle("hidden", !segments.length); $("#motionSeries").innerHTML = samples.map(item => `<span style="height:${Math.max(3, Number(item.motion || 0) * 100)}%"></span>`).join(""); $("#clipRows").innerHTML = segments.map(item => `<tr data-frame="${item.start_frame}"><td>${fmtTime(item.start_time)} – ${fmtTime(item.end_time)}</td><td><span class="state ${item.state}">${item.state === "valid" ? "有效操作" : item.state === "invalid" ? "无效片段" : "待确认"}</span></td><td>${(Number(item.confidence || 0) * 100).toFixed(1)}%</td><td>${esc(item.reason)}</td></tr>`).join(""); $$("#clipRows tr").forEach(row => row.addEventListener("click", () => updateFrame(Number(row.dataset.frame)))); updateBehaviorRemovalState(); refreshTimelineVisibility(); renderCurrentFrame(); }
  function clearCurationReport() { state.curation = null; state.curationStageFilter = null; $("#curationResult").classList.add("hidden"); $("#curationTrackGroup").classList.add("hidden"); $("#curationTrack").innerHTML = ""; $("#frameAuditFindings").classList.add("hidden"); refreshTimelineVisibility(); renderCurrentFrame(); }
  function curationTrackSegments(payload, stageId = null) {
    if (!stageId) return payload?.segments || [];
    const stage = (payload?.stages || []).find(item => item.id === stageId);
    if (["skipped", "not_evaluated"].includes(stage?.status)) return [];
    const frameCount = Number(payload?.source_video?.frame_count || state.media?.frame_count || 0), findings = (payload?.findings || []).filter(item => item.stage === stageId);
    if (!frameCount) return [];
    if (!findings.length) return [{ start_frame: 0, end_frame: Math.max(0, frameCount - 1), state: "valid", reason: "该阶段没有命中异常", confidence: 1 }];
    const boundaries = new Set([0, frameCount]); findings.forEach(item => { boundaries.add(Math.max(0, Number(item.start_frame))); boundaries.add(Math.min(frameCount, Number(item.end_frame) + 1)); }); const points = [...boundaries].sort((a, b) => a - b), segments = [];
    for (let index = 0; index < points.length - 1; index += 1) { const start = points[index], end = points[index + 1] - 1, hit = findings.filter(item => Number(item.start_frame) <= end && Number(item.end_frame) >= start), reject = hit.some(item => item.severity === "reject"); segments.push({ start_frame: start, end_frame: end, state: hit.length ? (reject ? "invalid" : "uncertain") : "valid", reason: qualityDisplayText(hit.map(item => item.reason).join("；")) || "该阶段未命中", confidence: hit.length ? Math.max(...hit.map(item => Number(item.confidence || 0))) : 1 }); }
    return segments;
  }
  function renderCurationTrack() {
    const payload = state.curation, track = $("#curationTrack"); if (!payload) { track.innerHTML = ""; return; }
    const activeStage = state.curationStageFilter ? (payload.stages || []).find(item => item.id === state.curationStageFilter) : null;
    const segments = curationTrackSegments(payload, state.curationStageFilter);
    const unavailable = activeStage && ["skipped", "not_evaluated"].includes(activeStage.status);
    track.innerHTML = unavailable
      ? `<span class="curation-track-empty" title="${escAttr(qualityDisplayText(activeStage.message) || "该阶段未执行")}"><b>未评估</b><span>${esc(qualityDisplayText(activeStage.message) || "缺少运行先决条件")}</span></span>`
      : segments.map(item => `<button type="button" class="segment ${item.state === "valid" ? "good" : item.state === "invalid" ? "bad" : "uncertain"}" data-frame="${Number(item.start_frame || 0)}" title="${escAttr(qualityDisplayText(item.reason))}" style="flex:${Math.max(1, Number(item.end_frame || 0) - Number(item.start_frame || 0) + 1)}"></button>`).join("") + (segments.length ? '<i class="timeline-cursor" aria-hidden="true"></i>' : "");
    $$("button.segment", track).forEach(button => button.addEventListener("click", () => updateFrame(Number(button.dataset.frame))));
    const filterButton = $("#clearCurationStageFilter"), filterName = qualityDisplayText(activeStage?.name) || "全部阶段";
    filterButton.textContent = state.curationStageFilter ? filterName : "全部阶段";
    filterButton.title = state.curationStageFilter ? `清除阶段筛选：${filterName}` : "显示全部质量阶段";
    updateTimelineCursor();
  }
  function renderCurationReport(payload) {
    state.curation = payload; state.curationStageFilter = null; $("#curationResult").classList.remove("hidden");
    const summary = payload.summary || {}, outdated = Boolean(payload.requires_rerun), recommendation = outdated ? "需重新运行" : ({ keep: "建议保留", review_and_apply: "需要审阅", exclude_episode: "建议排除 EP" }[summary.recommendation] || "待审阅");
    $("#curationRecommendation").textContent = recommendation; $("#curationRecommendation").className = `badge ${!outdated && summary.recommendation === "keep" ? "ready" : ""}`;
    $("#curationSummary").textContent = `${Number(summary.valid_frame_count || 0).toLocaleString()} 通过 · ${Number(summary.review_frame_count || 0).toLocaleString()} 待审 · ${Number(summary.invalid_frame_count || 0).toLocaleString()} 异常 · ${outdated ? "算法已升级，本报告仅供查看" : "源文件保持只读"}`;
    $("#curationStageList").innerHTML = (payload.stages || []).map(item => { const name = qualityDisplayText(item.name), message = qualityDisplayText(item.message) || item.status; return `<button type="button" class="curation-stage-row ${escAttr(item.status || "skipped")}" data-stage="${escAttr(item.id)}" title="${escAttr(message)}"><i></i><span><b>${esc(item.id.toUpperCase())} · ${esc(name)}</b><small>${esc(message)}</small></span></button>`; }).join("");
    $$(".curation-stage-row", $("#curationStageList")).forEach(button => button.addEventListener("click", () => { state.curationStageFilter = state.curationStageFilter === button.dataset.stage ? null : button.dataset.stage; $$(".curation-stage-row", $("#curationStageList")).forEach(item => item.classList.toggle("active", item.dataset.stage === state.curationStageFilter)); renderCurationTrack(); }));
    renderCurationTrack(); refreshTimelineVisibility(); renderCurrentFrame();
  }
  function currentSample() { return (state.annotations?.samples || []).reduce((best, item) => !best || Math.abs(Number(item.frame) - state.frame) < Math.abs(Number(best.frame) - state.frame) ? item : best, null); }
  function currentSegment() { return (state.annotations?.segments || []).find(item => state.frame >= item.start_frame && state.frame <= item.end_frame); }
  function currentCurationFindings() { return (state.curation?.findings || []).filter(item => state.frame >= Number(item.start_frame || 0) && state.frame <= Number(item.end_frame || 0)); }
  function currentCurationSegment() { return (state.curation?.segments || []).find(item => state.frame >= Number(item.start_frame || 0) && state.frame <= Number(item.end_frame || 0)); }
  function renderFrameAuditFindings() { const findings = currentCurationFindings(), segment = currentCurationSegment(), panel = $("#frameAuditFindings"); panel.classList.toggle("hidden", !state.curation); if (!state.curation) return; $("#frameAuditRows").innerHTML = findings.map(item => `<div class="frame-audit-row ${item.severity === "reject" ? "reject" : ""}"><b>${esc(String(item.stage || "").toUpperCase())}</b> · ${esc(qualityDisplayText(item.reason) || "质量候选")}</div>`).join("") || `<div class="frame-audit-row">${segment?.state === "valid" ? "当前帧通过已执行的质量检查" : esc(qualityDisplayText(segment?.reason) || "当前帧没有阶段命中")}</div>`; }
  function renderCurrentFrame() { renderFrameAuditFindings(); const sample = currentSample(), segment = currentSegment(), auditSegment = currentCurationSegment(); if (!sample && !segment) { clearCurrentResult(); $("#resultEmpty").classList.toggle("hidden", Boolean(state.curation)); const auditInvalid = auditSegment?.state === "invalid"; $("#invalidLabel").textContent = "DATA QUALITY"; $("#invalidLabel").classList.toggle("hidden", !auditInvalid); return; } const result = segment || sample; $("#resultEmpty").classList.add("hidden"); $("#resultBox").classList.remove("hidden"); const invalid = result.state === "invalid"; $("#resultLabel").textContent = result.state === "valid" ? "有效操作" : invalid ? "无效操作" : "待人工确认"; $("#resultDesc").textContent = result.reason || ""; $("#frameConfidence").textContent = `CONF ${Number(result.confidence || 0).toFixed(2)}`; metric("motion", sample?.motion ?? result.motion ?? 0); metric("contact", sample?.contact ?? result.contact ?? 0); metric("intent", result.confidence ?? 0); $("#invalidLabel").textContent = invalid ? "NO VALID ACTION" : "DATA QUALITY"; $("#invalidLabel").classList.toggle("hidden", !invalid && auditSegment?.state !== "invalid"); }
  function clearCurrentResult() { $("#resultEmpty").classList.remove("hidden"); $("#resultBox").classList.add("hidden"); $("#invalidLabel").classList.add("hidden"); }
  function metric(name, value) { const v = Math.max(0, Math.min(1, Number(value || 0))); $(`#${name}Value`).textContent = v.toFixed(2); $(`#${name}Bar`).style.width = `${v * 100}%`; }

  function renderSchema(profile) { const understanding = profile?.understanding; if (!understanding) { $("#schemaStatusBadge").textContent = profile?.status === "awaiting_vlm" ? "待 Qwen" : profile?.status === "error" ? "失败" : "未运行"; $("#schemaEmpty").classList.remove("hidden"); $("#schemaContent").classList.add("hidden"); return; } $("#schemaStatusBadge").textContent = "已理解"; $("#schemaEmpty").classList.add("hidden"); $("#schemaContent").classList.remove("hidden"); $("#schemaFormat").textContent = understanding.format_family || "unknown"; $("#schemaConfidence").textContent = `${Math.round(Number(understanding.format_confidence || 0) * 100)}%`; $("#inventoryFields").textContent = profile.inventory?.field_count || 0; $("#schemaSummary").textContent = understanding.summary || ""; $("#streamList").innerHTML = (understanding.streams || []).map(stream => { const path = `${stream.source_path || ""}${stream.field ? ` / ${stream.field}` : ""}`; return `<div class="stream-card"><b>${esc(stream.kind)} · ${esc(stream.modality)}</b><span>${esc(stream.side)} ·</span><span class="stream-path" title="${escAttr(path)}">${esc(path)}</span><span class="stream-confidence">conf ${(Number(stream.confidence || 0) * 100).toFixed(0)}%</span></div>`; }).join(""); $("#associationList").innerHTML = (understanding.associations || []).map(item => `<div class="association" title="${escAttr(item.vision_id)}"><b>${esc(item.side)} 视觉关联</b><br>${esc(item.vision_id)} → joints ${item.joint_ids?.length || 0} · sensors ${item.sensor_ids?.length || 0} · ${esc(item.time_alignment)}</div>`).join(""); $("#schemaWarnings").innerHTML = (profile.warnings || []).map(item => `<li>${esc(item)}</li>`).join(""); }

  function openExcludeFileModal() {
    const selection = state.treeSelection;
    if (!state.dataset || !selection?.fileIds?.length) { toast("请先在文件树中选择文件或 Episode", "error"); return; }
    const isEpisode = selection.kind === "episode";
    $("#excludeFileModalTitle").textContent = isEpisode ? "移出整个 Episode" : "移出文件";
    $("#confirmExcludeFileLabel").textContent = isEpisode ? `确认移出 EP（${selection.fileIds.length} 项）` : "确认移出文件";
    $("#excludeFileContext").innerHTML = isEpisode
      ? `<b>${esc(selection.label)}</b><br><span>${selection.fileIds.length} 个成员将从当前逻辑索引移出</span>`
      : `<b>${esc(selection.label)}</b><br><span>${esc(state.file?.relative_path || "")}</span>`;
    $("#excludeFileReason").value = isEpisode ? "人工排除不符合要求的 Episode" : "人工排除不符合条目"; $("#excludeFileModal").classList.remove("hidden"); lucide.createIcons();
  }
  function closeExcludeFileModal() { $("#excludeFileModal").classList.add("hidden"); }
  async function confirmExcludeFile() {
    const selection = state.treeSelection;
    if (!state.dataset || !selection?.fileIds?.length) return;
    const snapshot = { ...selection, fileIds: [...selection.fileIds] }, button = $("#confirmExcludeFile"); button.disabled = true;
    try {
      const result = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/exclusions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file_ids: snapshot.fileIds, reason: $("#excludeFileReason").value.trim() || "人工排除不符合条目", scope_type: snapshot.kind, scope_label: snapshot.label }) });
      closeExcludeFileModal(); renderDataset(result.dataset); toast(`已从数据集索引移出 ${snapshot.label}`, ""); setStatus(`已移出 ${snapshot.label} · ${snapshot.fileIds.length} 个条目 · 源文件未删除`);
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { button.disabled = false; }
  }

  async function openFolder() { const finishProgress = beginDatasetLoadProgress(); let progressFinished = false; try { const data = await api("/api/system/open-dataset-folder", { method: "POST" }); if (data.cancelled) { setStatus("已取消"); return; } if (data.mode === "collection") { finishProgress(); progressFinished = true; installDatasetCollection(data); setStatus(`发现 ${data.dataset_count} 个独立数据集 · 首个数据集按需加载`); await loadCollectionDataset(data.datasets[0].key); return; } state.datasetLoadToken += 1; state.datasetCollection = null; state.datasetCache = new Map(); renderDatasetSelector(); renderDataset(data.dataset); const schemaStatus = data.dataset?.schema_profile?.status; toast(schemaStatus === "completed" ? `已打开 ${data.dataset.name} · 格式已自动理解` : `已打开 ${data.dataset.name}`, schemaStatus === "error" ? "error" : ""); setStatus(schemaStatus === "completed" ? "数据集已打开 · Qwen 已自动理解格式与 Episode 归属" : schemaStatus === "error" ? `数据集已打开 · 自动格式理解失败：${data.dataset.schema_profile?.error || "未知错误"}` : "数据集已打开 · Qwen 未配置，格式理解等待再次运行"); } catch (error) { setStatus(error.message); toast(error.message, "error"); } finally { if (!progressFinished) finishProgress(); } }
  async function refreshDataset() { if (!state.dataset) return; const finishProgress = beginDatasetLoadProgress(); try { const data = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/rescan`, { method: "POST" }); renderDataset(data); const schemaStatus = data.schema_profile?.status; toast(schemaStatus === "completed" ? "扫描已刷新 · 格式已自动理解" : "扫描已刷新", schemaStatus === "error" ? "error" : ""); setStatus(schemaStatus === "completed" ? "原地重扫完成 · Qwen 已自动理解格式与 Episode 归属" : schemaStatus === "error" ? `原地重扫完成 · 自动格式理解失败：${data.schema_profile?.error || "未知错误"}` : "原地重扫完成 · 格式理解等待再次运行"); } catch (error) { toast(error.message, "error"); setStatus(error.message); } finally { finishProgress(); } }
  async function understandSchema() { if (!state.dataset) return; const finishProgress = beginQwenProgress(); try { const profile = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/analyze-schema`, { method: "POST" }); state.dataset.schema_profile = profile; if (profile.episode_resolution) state.dataset.episode_resolution = profile.episode_resolution; renderSchema(profile); renderResolvedTree(); setStatus("Qwen 已再次理解数据结构与 Episode 归属"); toast("格式与 Episode 归属已重新理解", ""); } catch (error) { $("#schemaStatusBadge").textContent = "失败"; toast(error.message, "error"); setStatus(error.message); } finally { finishProgress(); } }
  function manualRangeBounds() {
    const media = state.media || state.episode, fps = Number(media?.fps || 30), frameCount = Number(media?.frame_count || 0);
    const startFrame = Math.max(0, Math.min(frameCount - 1, Math.round(Number($("#manualRangeStart").value || 0) * fps)));
    const endFrame = Math.max(0, Math.min(frameCount - 1, Math.round(Number($("#manualRangeEnd").value || 0) * fps)));
    return { media, fps, frameCount, startFrame, endFrame };
  }
  function updateManualRangeSummary() {
    const { fps, startFrame, endFrame } = manualRangeBounds(), node = $("#manualRangeSummary");
    if (endFrame < startFrame) { node.classList.add("error"); node.textContent = "结束时间不能早于开始时间"; $("#saveManualRange").disabled = true; return; }
    node.classList.remove("error"); $("#saveManualRange").disabled = false;
    const stateLabel = $("input[name='manualRangeState']:checked")?.value === "valid" ? "有效操作" : "无效 / 无动作";
    node.textContent = `${stateLabel} · Frame ${startFrame + 1} – ${endFrame + 1} · 共 ${endFrame - startFrame + 1} 帧 / ${((endFrame - startFrame + 1) / fps).toFixed(3)} 秒`;
  }
  function openManualRange() {
    const media = state.media || state.episode;
    if (!state.dataset || !state.episode || !media?.frame_count) { toast("请先选择一个 Episode 视频", "error"); return; }
    const fps = Number(media.fps || 30), endFrame = Math.min(Number(media.frame_count) - 1, state.frame + Math.max(1, Math.round(fps)) - 1);
    $("#manualRangeStart").max = ((Number(media.frame_count) - 1) / fps).toFixed(3); $("#manualRangeEnd").max = $("#manualRangeStart").max;
    $("#manualRangeStart").value = (state.frame / fps).toFixed(3); $("#manualRangeEnd").value = (endFrame / fps).toFixed(3);
    $("#manualRangeContext").textContent = `${state.episode.name} / ${media.stream_name || "primary"} · ${Number(media.frame_count).toLocaleString()} 帧 · ${fps.toFixed(2)} FPS`;
    $("#manualRangeReason").value = "人工指定区间"; $("#manualRangeModal").classList.remove("hidden"); updateManualRangeSummary(); lucide.createIcons();
  }
  function closeManualRange() { $("#manualRangeModal").classList.add("hidden"); }
  function setManualRangeCurrent(target) { const media = state.media || state.episode, fps = Number(media?.fps || 30); $(target).value = (state.frame / fps).toFixed(3); updateManualRangeSummary(); }
  async function saveManualRange() {
    if (!state.dataset || !state.episode) return;
    const { media, startFrame, endFrame } = manualRangeBounds();
    if (endFrame < startFrame) { updateManualRangeSummary(); return; }
    const rangeState = $("input[name='manualRangeState']:checked")?.value || "invalid", button = $("#saveManualRange"); button.disabled = true;
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/segments`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start_frame: startFrame, end_frame: endFrame, state: rangeState, reason: $("#manualRangeReason").value.trim() || "人工指定区间", confidence: 1, media_file_id: media.file_id || null }) });
      state.annotations = payload; renderAnnotations(payload); await loadChangeCatalog(); closeManualRange(); updateFrame(startFrame);
      toast(`已暂存 ${endFrame - startFrame + 1} 帧${rangeState === "valid" ? "有效" : "无效"}区间`, ""); setStatus("人工区间标注已写入 .alicePD，源视频保持只读");
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { button.disabled = false; }
  }
  async function exportFolder() { if (!state.dataset) return; try { const data = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/export-folder-dialog?include_media=${$("#includeMedia").checked}`, { method: "POST" }); if (!data.cancelled) toast(`已导出到 ${data.path}`, ""); } catch (error) { toast(error.message, "error"); } }
  function downloadZip() { if (!state.dataset) return; window.open(`/api/datasets/${encodeURIComponent(state.dataset.id)}/export.zip?include_media=${$("#includeMedia").checked}`, "_blank"); }

  function showView(view) { state.view = view; $("#dashboardView").classList.toggle("active", view === "dashboard"); $("#reviewView").classList.toggle("active", view === "review"); $("#workspaceTitle").textContent = view === "dashboard" ? "数据集概览" : `Episode / ${state.episode?.name || "未选择"}`; }
  function configureModal() { $("#modelModal").classList.remove("hidden"); $("#modelModalStatus").textContent = "本地模型会在服务端加载并真实 warm-up；Qwen 会发送验证请求。"; }
  async function saveModel() { const type = $("#modelType").value; const payload = type === "qwen" ? { slot: "vlm", kind: "qwen", endpoint: $("#apiEndpoint").value.trim(), api_key: $("#apiKey").value.trim(), model: $("#qwenModel").value.trim(), verify: true } : { slot: "local", kind: type, model_path: $("#modelPath").value.trim(), device: $("#device").value, confidence: .25 }; $("#saveModel").disabled = true; $("#modelModalStatus").textContent = "正在加载/验证…"; try { await api("/api/models/configure", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); $("#modelModalStatus").textContent = "配置成功"; renderModels(); toast("模型状态已更新", ""); } catch (error) { $("#modelModalStatus").textContent = error.message; toast(error.message, "error"); } finally { $("#saveModel").disabled = false; } }

  const behaviorPhaseLabels = {
    approach: "接近 / Approach",
    observe: "观察 / Observe",
    reach: "伸手 / Reach",
    pre_grasp: "预抓取 / Pre-grasp",
    grasp: "抓取 / Grasp",
    contact: "接触 / Contact",
    lift: "抬起 / Lift",
    move: "移动 / Move",
    transport: "搬运 / Transport",
    align: "对齐 / Align",
    place: "放置 / Place",
    release: "释放 / Release",
    retract: "收回 / Retract",
    withdraw: "撤回 / Withdraw",
    manipulate: "操作 / Manipulate",
    inspect: "检查 / Inspect",
    idle: "静止 / Idle",
    unknown: "未知 / Unknown",
  };
  const behaviorPhaseTones = {
    idle: 5, observe: 0, reach: 0, grasp: 3, lift: 2, transport: 1,
    align: 4, place: 2, release: 3, withdraw: 5, manipulate: 4, inspect: 0, unknown: 5,
  };
  function behaviorPhaseTone(segment, fallbackIndex = 0) {
    const raw = behaviorReadableLabel(segment?.phase_label || segment?.phase || segment?.stage_label || segment?.stage || segment?.label);
    const key = raw.toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    return behaviorPhaseTones[key] ?? (fallbackIndex % 6);
  }
  function behaviorReadableLabel(value) {
    if (value == null) return "";
    if (typeof value !== "object") return String(value).trim();
    const zh = value.zh || value.zh_cn || value.cn || value.chinese || "";
    const en = value.en || value.en_us || value.english || "";
    if (zh && en && String(zh).trim().toLocaleLowerCase() !== String(en).trim().toLocaleLowerCase()) return `${String(zh).trim()} / ${String(en).trim()}`;
    return String(zh || en || value.label || value.name || value.value || value.source || value.method || "").trim();
  }
  function readableBehaviorPhase(segment) {
    const skill = behaviorReadableLabel(segment?.skill);
    if (skill) {
      const translated = behaviorReadableLabel(segment?.skill_zh);
      return translated ? `${translated} / ${skill}` : skill;
    }
    const preferred = behaviorReadableLabel(segment?.phase_label);
    const legacy = behaviorReadableLabel(segment?.phase || segment?.stage_label || segment?.stage || segment?.label);
    const raw = preferred || legacy;
    if (!raw) return "未提供";
    const key = raw.toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    return behaviorPhaseLabels[key] || raw;
  }
  function behaviorBoundarySource(segment, payload) {
    const raw = behaviorReadableLabel(segment?.boundary_source ?? segment?.boundary?.source ?? payload?.boundary_source ?? payload?.boundary?.source);
    if (!raw) return null;
    const key = raw.toLowerCase();
    if (key.includes("joint") || key.includes("fk") || raw.includes("校正")) return { label: "Joint 校正", kind: "joint", raw };
    if (key.includes("vlm") || key.includes("qwen") || key.includes("vision") || raw.includes("视觉语言")) return { label: "VLM", kind: "vlm", raw };
    return { label: raw, kind: "other", raw };
  }
  function behaviorTargetNames(values) {
    return (Array.isArray(values) ? values : []).map(item => behaviorReadableLabel(typeof item === "object" ? (item.name || item.label || item) : item)).filter(Boolean);
  }
  function activateInspector(name) {
    $$("[data-inspector]").forEach(item => item.classList.toggle("active", item.dataset.inspector === name));
    $$(".inspector-panel").forEach(panel => panel.classList.toggle("active", panel.id === `${name}Inspector`));
  }
  function clearBehaviorAnnotation() {
    state.behavior = null;
    $("#behaviorResult").classList.add("hidden"); $("#behaviorSegments").classList.add("hidden"); $("#behaviorMedium").classList.add("hidden"); $("#behaviorInspectorEmpty").classList.remove("hidden");
    $("#behaviorRemoveSelect").innerHTML = '<option value="">暂无可选动作</option>'; $("#behaviorRemoveSelect").disabled = true; $("#behaviorRemoveButton").disabled = true; $("#behaviorRemoveSummary").textContent = "请先运行 VLM 行为标注";
    refreshTimelineVisibility();
  }
  function behaviorPhaseOptions(segments) {
    const groups = new Map();
    for (const segment of segments) {
      const raw = behaviorReadableLabel(segment?.phase_label || segment?.phase || segment?.stage_label || segment?.stage || segment?.label);
      const key = raw.toLowerCase().replaceAll("-", "_").replaceAll(" ", "_"); if (!key) continue;
      const item = groups.get(key) || { key, label: readableBehaviorPhase(segment), segmentCount: 0, frameCount: 0 };
      item.segmentCount += 1; item.frameCount += Math.max(0, Number(segment.end_frame ?? segment.start_frame ?? 0) - Number(segment.start_frame || 0) + 1); groups.set(key, item);
    }
    return [...groups.values()];
  }
  function currentBehaviorRemovalOption() { return behaviorPhaseOptions(state.behavior?.segments || []).find(item => item.key === $("#behaviorRemoveSelect").value) || null; }
  function updateBehaviorRemovalState() {
    const button = $("#behaviorRemoveButton"), summary = $("#behaviorRemoveSummary"), option = currentBehaviorRemovalOption();
    const removals = Array.isArray(state.annotations?.behavior_removals) ? state.annotations.behavior_removals : [];
    const removed = option && removals.some(item => String(item.phase_label || "").toLowerCase().replaceAll("-", "_").replaceAll(" ", "_") === option.key && String(item.behavior_annotation_created_at || "") === String(state.behavior?.created_at || ""));
    button.disabled = !option || Boolean(removed);
    if (!option) { summary.textContent = "请选择一种动作"; return; }
    summary.textContent = removed
      ? `“${option.label}”已暂存为无效动作，等待应用`
      : `将去除 ${option.segmentCount} 个区间 · ${option.frameCount.toLocaleString()} 帧；其他动作保持不变`;
  }
  function renderBehaviorAnnotation(payload) {
    state.behavior = payload; $("#behaviorResult").classList.remove("hidden"); $("#behaviorInspectorEmpty").classList.add("hidden");
    const taskLabel = behaviorReadableLabel(payload.task_label) || behaviorReadableLabel(payload.label) || "other";
    $("#behaviorLabel").textContent = taskLabel; $("#behaviorLabel").title = `高层任务：${taskLabel}`;
    const direction = behaviorReadableLabel(payload.direction), directionLabels = { forward: "正向", reverse: "逆向", unknown: "方向未知" };
    $("#behaviorDirection").textContent = directionLabels[direction.toLowerCase()] || direction; $("#behaviorDirection").classList.toggle("hidden", !direction);
    $("#behaviorDescription").textContent = payload.behavior_description || payload.description || "";
    $("#behaviorConfidence").textContent = `CONF ${Number(payload.confidence || 0).toFixed(2)}`;
    const objectNouns = behaviorTargetNames(Array.isArray(payload.object_nouns) && payload.object_nouns.length ? payload.object_nouns : payload.primary_targets); $("#behaviorTargets").innerHTML = objectNouns.map(name => `<span class="behavior-target" title="VLM 语句物体名词">${esc(name)}</span>`).join("") || '<span class="empty-copy">未识别物体名词</span>';
    const medium = Array.isArray(payload.medium) ? payload.medium : []; $("#behaviorMedium").classList.toggle("hidden", !medium.length);
    $("#behaviorMediumRows").innerHTML = medium.map(item => { const start = Number(item.start_frame || 0), end = Number(item.end_frame ?? start); return `<button type="button" data-frame="${start}"><span>${esc(item.description || "未命名子任务")}</span><em>${start.toLocaleString()}–${end.toLocaleString()}</em></button>`; }).join("");
    $$("#behaviorMediumRows button").forEach(row => row.addEventListener("click", () => updateFrame(Number(row.dataset.frame))));
    const segments = Array.isArray(payload.segments) ? payload.segments : []; $("#behaviorSegments").classList.toggle("hidden", !segments.length);
    const options = behaviorPhaseOptions(segments), select = $("#behaviorRemoveSelect"), previous = select.value;
    select.innerHTML = options.length ? options.map(item => `<option value="${escAttr(item.key)}">${esc(item.label)} · ${item.segmentCount} 段</option>`).join("") : '<option value="">暂无可选动作</option>';
    select.disabled = !options.length; select.value = options.some(item => item.key === previous) ? previous : options[0]?.key || "";
    const fps = Math.max(0.01, Number(payload.source_video?.fps || state.media?.fps || state.episode?.fps || 30));
    $("#behaviorSegmentRows").innerHTML = segments.map(segment => {
      const startFrame = Number(segment.start_frame || 0), endFrame = Number(segment.end_frame ?? startFrame);
      const startTime = segment.start_time ?? startFrame / fps, endTime = segment.end_time ?? endFrame / fps;
      const phase = readableBehaviorPhase(segment), originalPhase = behaviorReadableLabel(segment.phase_label);
      const description = behaviorReadableLabel(segment.description) || behaviorReadableLabel(segment.label) || taskLabel;
      const targetInstance = behaviorReadableLabel(segment.target_instance);
      const targets = (behaviorTargetNames(segment.primary_targets).join(", ") || "—") + (targetInstance ? ` · ${targetInstance}` : "");
      const boundary = behaviorBoundarySource(segment, payload);
      const boundaryCell = boundary ? `<span class="behavior-boundary-source ${boundary.kind}" title="boundary_source: ${escAttr(boundary.raw)}">${esc(boundary.label)}</span>` : '<span class="behavior-boundary-source legacy" title="旧版标注未提供 boundary_source">边界未知</span>';
      return `<button type="button" class="behavior-segment-card" data-frame="${startFrame}"${boundary ? ` title="边界来源：${escAttr(boundary.label)}"` : ""}><span class="behavior-segment-card-head"><b class="behavior-phase-label" title="${escAttr(originalPhase ? `phase_label: ${originalPhase}` : "兼容旧版 label")}">${esc(phase)}</b><em>${fmtTime(startTime)} – ${fmtTime(endTime)}</em></span><span class="behavior-segment-description">${esc(description)}</span><span class="behavior-segment-meta"><i>${esc(targets)}</i>${boundaryCell}<i>${(Number(segment.confidence ?? payload.confidence ?? 0) * 100).toFixed(0)}%</i></span></button>`;
    }).join("");
    $$(".behavior-segment-card", $("#behaviorSegmentRows")).forEach(row => row.addEventListener("click", () => updateFrame(Number(row.dataset.frame))));
    updateBehaviorRemovalState(); refreshTimelineVisibility();
  }
  async function loadBehaviorAnnotation() {
    const datasetId = state.dataset?.id, episodeId = state.episode?.id, mediaFileId = state.media?.file_id || null, selectionToken = state.reviewSelectionToken;
    if (!datasetId || !episodeId) { clearBehaviorAnnotation(); return; }
    const isCurrent = () => isCurrentReviewSelection(datasetId, episodeId, mediaFileId, selectionToken);
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(episodeId)}/behavior-annotation`);
      if (!isCurrent()) return;
      if (payload.source_video?.file_id && mediaFileId && payload.source_video.file_id !== mediaFileId) { clearBehaviorAnnotation(); return; }
      renderBehaviorAnnotation(payload);
    }
    catch (_) { if (isCurrent()) clearBehaviorAnnotation(); }
  }
  async function removeBehaviorPhase() {
    if (!state.dataset || !state.episode || !state.behavior) { toast("请先选择 Episode 并运行 VLM 行为标注", "error"); return; }
    const option = currentBehaviorRemovalOption(), media = state.media || state.episode, button = $("#behaviorRemoveButton");
    if (!option) { toast("请选择要去除的动作", "error"); return; }
    button.disabled = true;
    try {
      const payload = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/behavior-removals`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ phase_label: option.key, media_file_id: media.file_id || null, reason: `按 VLM 动作去除：${option.label}` }) });
      state.annotations = payload; renderAnnotations(payload); updateBehaviorRemovalState(); await loadChangeCatalog();
      const removal = (payload.behavior_removals || []).at(-1); if (removal?.intervals?.length) updateFrame(Number(removal.intervals[0][0] || 0));
      toast(`已暂存去除“${option.label}” · ${Number(removal?.frame_count || option.frameCount).toLocaleString()} 帧`, ""); setStatus("按动作去除已写入 .alicePD，源数据保持只读");
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { updateBehaviorRemovalState(); }
  }
  function closeAnalysisScope() { $("#analysisScopeModal").classList.add("hidden"); const target = state.analysisReturnFocus; state.analysisReturnFocus = null; if (target?.isConnected) target.focus(); }
  function configureAnalysisForceOption() {
    let field = $("#analysisForceField");
    if (!field) {
      field = document.createElement("label");
      field.id = "analysisForceField";
      field.className = "analysis-scope-option hidden";
      field.innerHTML = '<input type="checkbox" id="forceAnalysis"><span><b>强制重新标注</b><small>忽略已有有效结果并重新调用 Qwen</small></span>';
      $("#analysisScopeNotice").before(field);
    }
    const behavior = state.analysisOperation === "vlm_behavior" || state.analysisOperation === "full_pipeline";
    field.classList.toggle("hidden", !behavior);
    if (!behavior) $("#forceAnalysis").checked = false;
  }
  function ensureCurationQualityGapControl() {
    if ($("#curationQualityGap")) return;
    const field = document.createElement("label");
    field.innerHTML = '低质量合并间隔（秒）<input id="curationQualityGap" type="number" min="0" max="2" step="0.05" value="0.3" title="只合并间隔严格小于此值的低质量标记">';
    const staticField = $("#curationStaticDuration")?.closest("label");
    if (staticField) staticField.before(field);
  }
  function ensureS1RepairControls() {
    if ($("#curationRepairS1")) return;
    const sigmaField = $("#curationSuddenSigma")?.closest("label");
    if (!sigmaField) return;
    const enabled = document.createElement("label");
    enabled.className = "analysis-scope-option";
    enabled.innerHTML = '<input type="checkbox" id="curationRepairS1" checked><span><b>修复孤立 S1 尖峰</b><small>只修复两侧有可靠锚点的短尖峰，源文件保持只读</small></span>';
    const maximum = document.createElement("label");
    maximum.innerHTML = '最大修复帧数<input id="curationS1MaxRepairFrames" type="number" min="1" max="15" step="1" value="5">';
    sigmaField.after(enabled, maximum);
  }
  function ensureFullActionControls() {
    if ($("#fullActionSettings")) return;
    const section = document.createElement("section");
    section.id = "fullActionSettings";
    section.className = "full-action-settings hidden";
    section.innerHTML = '<label>输出格式<select id="fullOutputFormat"><option value="lerobot" selected>LeRobot · 双手 21 点 + Body</option><option value="hdf5_mp4">HDF5 + MP4 · 旧版兼容</option></select><small class="field-hint">LeRobot 为默认；左右手各固定 21 点，其余命名关节写入 Body Parquet。</small></label><label class="analysis-scope-option full-action-toggle"><input type="checkbox" id="fullGenerateAction"><span><b>生成机器人 Action</b><small>可选；开启后为所有 Full 分片使用同一种机器人映射</small></span></label><div class="action-settings-grid hidden" id="fullActionFields"><label>机器人 / 控制类型<select id="fullRobotProfile"></select></label><label id="fullSourceHandField">映射哪只手<select id="fullSourceHand"><option value="right" selected>右手</option><option value="left">左手</option></select></label><label>统一坐标系<select id="fullCoordinateFrame"><option value="camera" selected>当前相机坐标（推荐）</option><option value="world">源数据世界坐标</option></select></label><label>预测未来帧数<input id="fullHorizonFrames" type="number" min="1" max="30" step="1" value="3"></label></div><div class="action-profile-note hidden" id="fullActionProfileNote"></div>';
    $("#curationPlan").append(section);
  }
  function trimNumber(id, fallback, scale = 1) {
    const input = $("#" + id), value = Number(input?.value);
    return Number.isFinite(value) ? value / scale : fallback;
  }
  function updateTrimSettingOutputs() {
    $("#yoloProximityValue").textContent = trimNumber("yoloProximityThreshold", 0.04, 1000).toFixed(3);
    $("#yoloMaxGapValue").textContent = `${trimNumber("yoloMaxGapSeconds", 0.5, 10).toFixed(1)} s`;
    $("#yoloMinValidValue").textContent = `${trimNumber("yoloMinValidSeconds", 0.3, 10).toFixed(1)} s`;
    $("#qwenTrimConfidenceValue").textContent = trimNumber("qwenTrimConfidence", 0.55, 100).toFixed(2);
    $("#qwenTrimMaxGapValue").textContent = `${trimNumber("qwenTrimMaxGapSeconds", 1.5, 10).toFixed(1)} s`;
    $("#qwenTrimMinValidValue").textContent = `${trimNumber("qwenTrimMinValidSeconds", 0.75, 20).toFixed(2)} s`;
  }
  function configureTrimSettings() {
    const yolo = state.analysisOperation === "no_action_trim", qwen = state.analysisOperation === "qwen_trim", visible = yolo || qwen;
    $("#trimSettings").classList.toggle("hidden", !visible);
    $("#yoloTrimSettings").classList.toggle("hidden", !yolo);
    $("#qwenTrimSettings").classList.toggle("hidden", !qwen);
    $("#trimSettingsSummary").textContent = yolo ? "YOLOE 距离与时序滤波" : qwen ? "Qwen 窗口判定与时序滤波" : "仅影响本次任务";
    if (visible) $("#trimSettings").open = true;
    updateTrimSettingOutputs();
  }
  function trimRequestConfig() {
    if (state.analysisOperation === "no_action_trim") return {
      sample_fps: trimNumber("yoloTrimSampleFps", 4),
      proximity_threshold: trimNumber("yoloProximityThreshold", 0.04, 1000),
      max_gap_seconds: trimNumber("yoloMaxGapSeconds", 0.5, 10),
      min_valid_seconds: trimNumber("yoloMinValidSeconds", 0.3, 10),
    };
    if (state.analysisOperation === "qwen_trim") return {
      sample_fps: trimNumber("qwenTrimSampleFps", 0.75),
      window_seconds: trimNumber("qwenTrimWindowSeconds", 1.5),
      frames_per_window: Math.round(trimNumber("qwenTrimFramesPerWindow", 2)),
      windows_per_request: Math.round(trimNumber("qwenTrimWindowsPerRequest", 10)),
      confidence_threshold: trimNumber("qwenTrimConfidence", 0.55, 100),
      max_gap_seconds: trimNumber("qwenTrimMaxGapSeconds", 1.5, 10),
      min_valid_seconds: trimNumber("qwenTrimMinValidSeconds", 0.75, 20),
    };
    return {};
  }
  function curationRequestConfig() { return {
    sudden_change_sigma: trimNumber("curationSuddenSigma", 6),
    repair_s1_spikes: $("#curationRepairS1")?.checked !== false,
    s1_max_repair_frames: Math.round(trimNumber("curationS1MaxRepairFrames", 5)),
    directional_agreement_threshold: trimNumber("curationDaThreshold", 0.65),
    max_lag_seconds: trimNumber("curationMaxLag", 0.5),
    outlier_alpha: trimNumber("curationOutlierAlpha", 0.1),
    video_sample_fps: trimNumber("curationVideoFps", 2),
    black_level_threshold: trimNumber("curationBlackThreshold", 8),
    blur_laplacian_threshold: trimNumber("curationBlurThreshold", 35),
    static_difference_threshold: 1.5,
    static_duration_seconds: trimNumber("curationStaticDuration", 2),
    quality_gap_merge_seconds: trimNumber("curationQualityGap", 0.3),
  }; }
  function fullActionRequestConfig() {
    const output = { full_output_format: $("#fullOutputFormat")?.value || "lerobot" };
    if (state.analysisOperation !== "full_pipeline" || !$("#fullGenerateAction")?.checked) return output;
    const profileId = $("#fullRobotProfile").value;
    const profile = state.actionProfiles.find(item => item.id === profileId);
    if (!profile) { toast("请选择有效的机器人类型", "error"); $("#fullRobotProfile").focus(); return null; }
    const horizon = Number($("#fullHorizonFrames").value);
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > 30) { $("#fullHorizonFrames").reportValidity(); return null; }
    return {
      ...output,
      full_action_profile_id: profileId,
      full_action_source_hand: $("#fullSourceHand").value,
      full_action_coordinate_frame: $("#fullCoordinateFrame").value,
      full_action_horizon_frames: horizon,
    };
  }
  function validateTrimSettings() {
    const section = ["paper_curation", "full_pipeline"].includes(state.analysisOperation) ? $("#curationPlan") : state.analysisOperation === "no_action_trim" ? $("#yoloTrimSettings") : state.analysisOperation === "qwen_trim" ? $("#qwenTrimSettings") : null;
    if (!section) return true;
    const invalid = [...section.querySelectorAll("input,select")].find(input => !input.checkValidity() || (input.type === "number" && !input.value.trim()));
    if (!invalid) return true;
    invalid.reportValidity(); invalid.focus(); toast(["paper_curation", "full_pipeline"].includes(state.analysisOperation) ? "请先修正清洗参数" : "请先修正剪切参数", "error");
    return false;
  }
  function analysisNeedsMedia() { return state.analysisOperation === "paper_curation" || state.analysisOperation === "full_pipeline" || state.analysisOperation === "video_smoothing" || state.analysisOperation === "no_action_trim" || state.analysisOperation === "qwen_trim"; }
  const curationStageOrder = ["s1", "s2", "s3", "s4", "s5", "c3", "c1", "c2"];
  const curationStageNames = { s1: "突变与 Jerk", s2: "State-Action 对齐与导出一致性", s3: "分位极值", s4: "FK 一致性", s5: "基座与方向统一", c1: "指令一致性", c2: "视频-State 一致性", c3: "视频质量与整手可见" };
  const curationStatusLabels = { ready: "可运行", pending: "等待前序", reused: "复用", skipped: "跳过", completed: "完成", warning: "复核", not_evaluated: "未评估" };
  function renderCurationPreflight(payload) {
    const stages = payload?.stages || [];
    $("#curationPlanStages").innerHTML = stages.map((item, index) => { const name = qualityDisplayText(item.name), message = qualityDisplayText(item.message); return `<div class="curation-plan-stage ${escAttr(item.status || "skipped")}"><span class="curation-stage-index">${index + 1}</span><span><b>${esc(String(item.id || "").toUpperCase())} · ${esc(name)}</b><small title="${escAttr(message)}">${esc(message)}</small></span><em>${esc(item.status_label || curationStatusLabels[item.status] || item.status || "--")}</em></div>`; }).join("");
    if (payload?.scope === "all") {
      const checked = Number(payload.checked_count || 0), total = Number(payload.episode_count || 0), pending = Number(payload.pending_media_count || 0), failed = Number(payload.failed_count || 0);
      $("#curationPreflightSummary").textContent = `逐 EP ${checked}/${total} 已检查${pending ? ` · ${pending} 待选视频` : ""}${failed ? ` · ${failed} 检查失败` : ""}`;
      return;
    }
    const runnable = stages.filter(item => ["ready", "reused"].includes(item.status)).length;
    $("#curationPreflightSummary").textContent = `${runnable}/${stages.length || 8} 可执行或复用 · 其余阶段明确跳过`;
  }
  function aggregateCurationPreflights(results, episodeCount, pendingMediaCount, failedCount) {
    const uncheckedCount = Math.max(0, episodeCount - results.length);
    const stages = curationStageOrder.map(stageId => {
      const items = results.map(payload => (payload.stages || []).find(item => item.id === stageId)).filter(Boolean);
      const exemplar = items[0] || {};
      const counts = items.reduce((summary, item) => { const key = item.status || "not_evaluated"; summary[key] = (summary[key] || 0) + 1; return summary; }, {});
      const parts = Object.entries(counts).map(([status, count]) => `${count} ${curationStatusLabels[status] || status}`);
      if (uncheckedCount) parts.push(`${uncheckedCount} 未检查`);
      const statuses = items.map(item => item.status || "not_evaluated"), unique = new Set(statuses);
      let status = "warning", statusLabel = "混合";
      if (!items.length) { status = "not_evaluated"; statusLabel = "未检查"; }
      else if (statuses.every(value => ["skipped", "not_evaluated"].includes(value))) { status = "skipped"; statusLabel = "全跳过"; }
      else if (uncheckedCount || statuses.some(value => ["warning", "pending", "skipped", "not_evaluated"].includes(value)) || unique.size > 1) { status = "warning"; statusLabel = "混合"; }
      else if (statuses.every(value => value === "reused")) { status = "reused"; statusLabel = "全复用"; }
      else if (statuses.every(value => value === "completed")) { status = "completed"; statusLabel = "已完成"; }
      else { status = "ready"; statusLabel = "可运行"; }
      return { id: stageId, name: exemplar.name || curationStageNames[stageId] || stageId.toUpperCase(), status, status_label: statusLabel, message: parts.join(" · ") || "未检查" };
    });
    return { scope: "all", stages, episode_count: episodeCount, checked_count: results.length, pending_media_count: pendingMediaCount, failed_count: failedCount };
  }
  async function updateCurationPreflight() {
    const plan = $("#curationPlan"), active = ["paper_curation", "full_pipeline"].includes(state.analysisOperation); plan.classList.toggle("hidden", !active); if (!active || !state.dataset) return;
    const token = ++state.curationPreflightToken, datasetId = state.dataset.id, all = $("input[name='analysisScope']:checked")?.value === "all", episodes = state.dataset.episodes || [];
    const isCurrent = () => token === state.curationPreflightToken && state.dataset?.id === datasetId && ["paper_curation", "full_pipeline"].includes(state.analysisOperation);
    $("#curationPreflightSummary").textContent = all ? `正在逐 EP 检查 0/${episodes.length}` : "正在检查真实先决条件";
    $("#curationPlanStages").innerHTML = curationStageOrder.map((id, index) => `<div class="curation-plan-stage"><span class="curation-stage-index">${index + 1}</span><span><b>${id.toUpperCase()} · ${esc(curationStageNames[id])}</b><small>检查中</small></span><em>...</em></div>`).join("");
    if (!all) {
      const episodeId = $("#analysisEpisodeSelect").value, mediaFileId = $("#analysisMediaSelect").value;
      if (!episodeId || !mediaFileId) { $("#curationPreflightSummary").textContent = "请先选择 Episode 与视频流"; $("#curationPlanStages").innerHTML = '<div class="analysis-media-map-empty">选择视频后将检查八个阶段的真实先决条件。</div>'; return; }
      try { const payload = await api(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(episodeId)}/curation-preflight?media_file_id=${encodeURIComponent(mediaFileId)}`); if (isCurrent()) renderCurationPreflight(payload); }
      catch (error) { if (isCurrent()) { $("#curationPreflightSummary").textContent = "先决条件检查失败"; $("#curationPlanStages").innerHTML = `<div class="analysis-media-map-empty">${esc(error.message)}</div>`; } }
      return;
    }
    const targets = [], pending = [];
    for (const episode of episodes) {
      const streams = episode.media_streams || [];
      const mapSelect = $$(".analysis-media-map-select", $("#analysisMediaMapList")).find(item => item.dataset.episodeId === episode.id);
      const mediaFileId = streams.length === 1 ? streams[0].file_id : mapSelect?.value || null;
      if (mediaFileId) targets.push({ episodeId: episode.id, mediaFileId }); else pending.push(episode.id);
    }
    if (!targets.length) { if (isCurrent()) renderCurationPreflight(aggregateCurationPreflights([], episodes.length, pending.length, 0)); return; }
    const results = [], failures = [];
    let cursor = 0, completed = 0;
    const worker = async () => {
      while (isCurrent()) {
        const index = cursor; cursor += 1;
        if (index >= targets.length) return;
        const target = targets[index];
        try { results.push(await api(`/api/datasets/${encodeURIComponent(datasetId)}/episodes/${encodeURIComponent(target.episodeId)}/curation-preflight?media_file_id=${encodeURIComponent(target.mediaFileId)}`)); }
        catch (error) { failures.push({ episode_id: target.episodeId, error: error.message }); }
        finally { completed += 1; if (isCurrent()) $("#curationPreflightSummary").textContent = `正在逐 EP 检查 ${completed}/${targets.length}${pending.length ? ` · ${pending.length} 待选视频` : ""}`; }
      }
    };
    await Promise.all(Array.from({ length: Math.min(6, targets.length) }, () => worker()));
    if (isCurrent()) renderCurationPreflight(aggregateCurationPreflights(results, episodes.length, pending.length, failures.length));
  }
  function mediaStreamName(media) { return String(media?.stream_name || media?.relative_path?.split("/").pop() || "video"); }
  function renderAnalysisMediaMap() {
    const map = $("#analysisMediaMap"), list = $("#analysisMediaMapList");
    const all = $("input[name='analysisScope']:checked")?.value === "all";
    const visible = analysisNeedsMedia() && all;
    map.classList.toggle("hidden", !visible);
    if (!visible) { list.innerHTML = ""; return; }
    const episodes = state.dataset?.episodes || [];
    const ambiguous = episodes.filter(item => (item.media_streams || []).length !== 1);
    const automatic = episodes.length - ambiguous.length;
    const missing = ambiguous.filter(item => !(item.media_streams || []).length).length;
    $("#analysisMediaMapSummary").textContent = `${automatic.toLocaleString()} 个单视频 EP 已锁定 · ${Math.max(0, ambiguous.length - missing).toLocaleString()} 个多视频 EP 需确认${missing ? ` · ${missing} 个缺少视频` : ""}`;
    list.innerHTML = ambiguous.map(episode => {
      const streams = episode.media_streams || [];
      if (!streams.length) return `<div class="analysis-media-map-row missing"><b title="${escAttr(episode.name)}">${esc(episode.name)}</b><span>没有可用视频流</span></div>`;
      return `<label class="analysis-media-map-row"><b title="${escAttr(episode.name)}">${esc(episode.name)}</b><select class="analysis-media-map-select" data-episode-id="${escAttr(episode.id)}"><option value="">请选择视频流</option>${streams.map(item => `<option value="${escAttr(item.file_id)}">${esc(mediaStreamName(item))} · ${Number(item.frame_count || 0).toLocaleString()} frames · ${Number(item.fps || 0).toFixed(2)} FPS</option>`).join("")}</select></label>`;
    }).join("") || '<div class="analysis-media-map-empty">所有 Episode 均只有一个视频流，已逐 EP 自动锁定。</div>';
    $$(".analysis-media-map-select", list).forEach(select => select.addEventListener("change", updateCurationPreflight));
  }
  function fillAnalysisMediaByName() {
    const referenceEpisode = state.dataset?.episodes?.find(item => item.id === $("#analysisEpisodeSelect").value);
    const referenceMedia = (referenceEpisode?.media_streams || []).find(item => item.file_id === $("#analysisMediaSelect").value);
    if (!referenceMedia) { toast("请先选择参考视频流", "error"); return; }
    const wanted = mediaStreamName(referenceMedia).toLocaleLowerCase();
    let matched = 0;
    $$(".analysis-media-map-select", $("#analysisMediaMapList")).forEach(select => {
      const episode = state.dataset.episodes.find(item => item.id === select.dataset.episodeId);
      const candidates = (episode?.media_streams || []).filter(item => mediaStreamName(item).toLocaleLowerCase() === wanted);
      if (candidates.length === 1) { select.value = candidates[0].file_id; matched += 1; }
    });
    toast(`已按参考流名填充 ${matched} 个多视频 Episode`, matched ? "" : "error"); updateCurationPreflight();
  }
  function updateAnalysisMediaOptions() {
    const field = $("#analysisMediaField"), select = $("#analysisMediaSelect"), needsMedia = analysisNeedsMedia();
    field.classList.toggle("hidden", !needsMedia);
    if (!needsMedia) { select.innerHTML = ""; return; }
    const episode = state.dataset?.episodes?.find(item => item.id === $("#analysisEpisodeSelect").value);
    const streams = episode?.media_streams || [];
    const previous = select.value;
    select.innerHTML = streams.map(item => `<option value="${escAttr(item.file_id)}">${esc(mediaStreamName(item))} · ${Number(item.frame_count || 0).toLocaleString()} frames · ${Number(item.fps || 0).toFixed(2)} FPS</option>`).join("");
    const preferred = state.episode?.id === episode?.id ? state.media?.file_id : episode?.primary_media_file_id;
    select.value = streams.some(item => item.file_id === previous) ? previous : streams.some(item => item.file_id === preferred) ? preferred : streams[0]?.file_id || "";
    const all = $("input[name='analysisScope']:checked")?.value === "all";
    $("#analysisMediaHint").textContent = all ? "单视频 Episode 自动锁定；多视频 Episode 必须在下方逐条确认。" : "任务只读取此视频，结果与源文件路径会写入 .alicePD。";
    renderAnalysisMediaMap(); updateCurationPreflight();
  }
  function updateAnalysisScope() {
    const all = $("input[name='analysisScope']:checked")?.value === "all";
    const episodes = state.dataset?.episodes || [];
    const fullOutputPath = `${state.dataset?.root_path || "数据集目录"}\\output`;
    $("#analysisEpisodeSelect").disabled = all;
    updateAnalysisMediaOptions();
    const fullProfile = state.actionProfiles.find(item => item.id === $("#fullRobotProfile")?.value);
    const fullActionText = $("#fullGenerateAction")?.checked && fullProfile ? `将生成 ${fullProfile.name} Action` : "不生成派生 Action；已有 Action 可用于 S2";
    const fullOutputText = $("#fullOutputFormat")?.value === "hdf5_mp4" ? "HDF5 + MP4 兼容格式" : "LeRobot 双手 21 点 + Body 格式";
    $("#analysisScopeNotice").textContent = state.analysisOperation === "full_pipeline"
      ? (all ? `将为 ${episodes.length} 个 Episode 执行视频平滑与清洗；${fullActionText}，再标注非红片段、执行 C1/C2，并以 ${fullOutputText} 输出到 ${fullOutputPath}。` : `视频平滑 → S1-S5/C3（${fullActionText}）→ 非红片段 VLM → C1/C2 → 去除静止和伸手 → 以 ${fullOutputText} 输出到 ${fullOutputPath}。`)
      : state.analysisOperation === "paper_curation"
      ? (all ? `将按所选视频流依次执行 S1-S5 与 C3 初筛，仅对有效片段调用 VLM，最后执行 C2；共 ${episodes.length} 个 Episode。` : "顺序：S1-S5 → C3 → 仅标注有效片段 → C2；所有报告先写入 .alicePD，源文件不变。")
      : state.analysisOperation === "video_smoothing"
      ? (all ? `将按所选同名视频流依次平滑 ${episodes.length} 个 Episode，输出只写入 .alicePD。` : "将执行光流稳像、边缘补偿和轻度锐化；它不能恢复曝光期间已经丢失的细节。")
      : state.analysisOperation === "vlm_behavior"
      ? (all ? `将检查 ${episodes.length} 个 Episode；已有 v3 分层行为标注与目标词文件的条目会直接复用，其余条目才调用 Qwen-VLM。` : "按高层任务与可变长度细阶段标注，再用 Joint 校正边界；已有有效 v3 结果会直接复用。")
      : state.analysisOperation === "no_action_trim"
        ? (all ? `将用各 Episode 的主要词文件运行 YOLOE；缺少词文件的 Episode 会报告失败。` : "将把主要词文件注入 YOLOE，按手/夹爪与物体距离生成有效区间并过滤短暂漏检。")
        : state.analysisOperation === "qwen_trim"
          ? (all ? `将对 ${episodes.length} 个 Episode 的指定视频流逐一调用 Qwen，独立生成有效 / 无效片段。` : "将按时间窗口抽帧调用 Qwen，对有效操作、等待、空闲和无关运动进行独立判段。")
        : (all ? `将检查 ${episodes.length} 个 Episode；已有结果或没有 mocap 的条目会自动跳过。` : "将检查所选 Episode 的 mocap 缺帧并运行 SLAM/VO 恢复。");
  }
  function openAnalysisScope(operation, trigger = document.activeElement) {
    if (!state.dataset?.episodes?.length) { toast("当前数据集没有可分析的 Episode", "error"); return; }
    state.analysisOperation = operation;
    state.analysisReturnFocus = trigger;
    configureAnalysisForceOption();
    configureTrimSettings();
    configureFullActionSettings();
    const full = operation === "full_pipeline", curation = operation === "paper_curation", smoothing = operation === "video_smoothing", behavior = operation === "vlm_behavior", trimming = operation === "no_action_trim", qwenTrimming = operation === "qwen_trim", episodes = state.dataset.episodes;
    $("#curationPlan").classList.toggle("hidden", !(curation || full));
    $("#analysisScopeTitle").textContent = full ? "Full 标准数据集范围" : curation ? "数据质量清洗范围" : smoothing ? "视频平滑范围" : behavior ? "VLM 行为标注范围" : trimming ? "YOLOE 无动作剪切范围" : qwenTrimming ? "Qwen 片段剪切范围" : "SLAM 位姿恢复范围";
    $("#analysisScopeKind").textContent = full ? "平滑 → S1-S5/C3（Action 可选）→ VLM → C1/C2 → LeRobot / 兼容格式导出" : curation ? "S1-S5 → C3 → 非红片段 VLM → C1/C2" : smoothing ? "光流稳像与轻度运动模糊缓解" : behavior ? "Qwen-VLM 高层任务 + 可变长度细阶段 + Joint 边界校正" : trimming ? "YOLOE 物体与手/夹爪距离分析" : qwenTrimming ? "Qwen-VLM 有效操作 / 无效片段判定" : "SLAM / Visual Odometry 初始位姿恢复";
    $("#analysisAllDescription").textContent = `依次处理全部 ${episodes.length} 个 Episode`;
    $("#analysisEpisodeSelect").innerHTML = episodes.map(item => `<option value="${escAttr(item.id)}">${esc(item.name)} · ${Number(item.frame_count || 0).toLocaleString()} frames</option>`).join("");
    $("#analysisEpisodeSelect").value = state.episode?.id || episodes[0].id;
    const single = $("input[name='analysisScope'][value='single']"); if (single) single.checked = true;
    updateAnalysisScope(); $("#analysisScopeModal").classList.remove("hidden"); lucide.createIcons(); requestAnimationFrame(() => $("#analysisScopeTitle").focus());
  }
  async function loadActionProfiles() {
    if (!state.actionProfiles.length) {
      try { const payload = await api("/api/action-mappings/profiles"); state.actionProfiles = payload.items || []; }
      catch (_) { state.actionProfiles = []; }
    }
    if (state.actionProfiles.length) for (const id of ["actionRobotProfile", "fullRobotProfile"]) {
      const select = $("#" + id); if (!select) continue;
      const previous = select.value;
      select.innerHTML = state.actionProfiles.map(item => `<option value="${escAttr(item.id)}">${esc(item.name)}</option>`).join("");
      select.value = state.actionProfiles.some(item => item.id === previous) ? previous : state.actionProfiles[0].id;
    }
    renderActionProfileNote();
    renderFullActionSettings();
  }
  function currentActionProfile() { return state.actionProfiles.find(item => item.id === $("#actionRobotProfile").value) || null; }
  function renderActionProfileNote() {
    const profile = currentActionProfile(), option = $("#actionRobotProfile").selectedOptions[0];
    const single = profile ? Number(profile.sides || 0) === 1 : /单臂/.test(option?.textContent || "");
    $("#actionSourceHandField").classList.toggle("hidden", !single);
    const description = profile?.description || "从腕部轨迹生成机器人末端 Action。";
    const caution = profile?.requires_ik ? " 输出是末端代理 Action；加载该机器人 URDF、关节限位和标定后才能转换为原生关节控制。" : " 输出可用于笛卡尔目标策略，部署前仍需机器人基座与尺度标定。";
    $("#actionProfileNote").textContent = `${description}${caution}`;
    updateActionMappingScope();
  }
  function configureFullActionSettings() {
    const section = $("#fullActionSettings"), full = state.analysisOperation === "full_pipeline";
    section.classList.toggle("hidden", !full);
    if (full) loadActionProfiles(); else renderFullActionSettings();
  }
  function renderFullActionSettings() {
    const section = $("#fullActionSettings"); if (!section) return;
    const full = state.analysisOperation === "full_pipeline", toggle = $("#fullGenerateAction"), available = state.actionProfiles.length > 0;
    const enabled = full && available && Boolean(toggle.checked), profile = state.actionProfiles.find(item => item.id === $("#fullRobotProfile").value) || null;
    toggle.disabled = full && !available;
    $("#fullActionFields").classList.toggle("hidden", !enabled);
    $("#fullActionProfileNote").classList.toggle("hidden", !enabled);
    $$("input,select", $("#fullActionFields")).forEach(input => { input.disabled = !enabled; });
    $("#fullSourceHandField").classList.toggle("hidden", enabled && Number(profile?.sides || 0) !== 1);
    if (enabled) {
      const requirement = profile?.requires_ik ? "输出为末端代理 Action，部署前仍需机器人标定与 IK。" : "输出为笛卡尔目标 Action，部署前仍需基座与尺度标定。";
      $("#fullActionProfileNote").textContent = `${profile?.description || "从腕部轨迹生成机器人 Action。"}${requirement}`;
    }
    if (full) updateAnalysisScope();
  }
  function updateActionMappingScope() {
    const all = $("input[name='actionScope']:checked")?.value === "all", episodes = state.dataset?.episodes || [], profile = currentActionProfile();
    $("#actionEpisodeSelect").disabled = all;
    const target = profile?.name || $("#actionRobotProfile").selectedOptions[0]?.textContent || "所选机器人";
    $("#actionMappingNotice").textContent = `${all ? `将处理全部 ${episodes.length} 个 Episode` : "只处理所选 Episode"} · ${target} · 每个成功结果都会立即写入 .alicePD/actions 并更新索引。`;
  }
  function openActionMapping(trigger = document.activeElement) {
    const episodes = state.dataset?.episodes || [];
    if (!episodes.length) { toast("当前数据集没有可生成 Action 的 Episode", "error"); return; }
    state.actionMappingReturnFocus = trigger;
    $("#actionEpisodeSelect").innerHTML = episodes.map(item => `<option value="${escAttr(item.id)}">${esc(item.name)} · ${Number(item.frame_count || 0).toLocaleString()} frames</option>`).join("");
    $("#actionEpisodeSelect").value = state.episode?.id || episodes[0].id;
    const single = $("input[name='actionScope'][value='single']"); if (single) single.checked = true;
    $("#actionAllDescription").textContent = `依次处理全部 ${episodes.length} 个 Episode`;
    $("#actionMappingModal").classList.remove("hidden"); loadActionProfiles(); updateActionMappingScope(); lucide.createIcons(); requestAnimationFrame(() => $("#actionMappingTitle").focus());
  }
  function closeActionMapping() { $("#actionMappingModal").classList.add("hidden"); const target = state.actionMappingReturnFocus; state.actionMappingReturnFocus = null; if (target?.focus) target.focus(); }
  async function submitActionMapping() {
    if (!state.dataset) return;
    const horizon = Number($("#actionHorizonFrames").value);
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > 30) { $("#actionHorizonFrames").reportValidity(); return; }
    const all = $("input[name='actionScope']:checked")?.value === "all";
    const episodeIds = all ? state.dataset.episodes.map(item => item.id) : [$("#actionEpisodeSelect").value];
    const button = $("#startActionMapping"); button.disabled = true;
    try {
      const payload = {
        episode_ids: episodeIds,
        profile_id: $("#actionRobotProfile").value,
        source_hand: $("#actionSourceHand").value,
        coordinate_frame: $("#actionCoordinateFrame").value,
        horizon_frames: horizon,
        force: false,
      };
      const job = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/action-jobs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      state.analysisJobs.set(job.id, job); renderAnalysisThreadStatus(); closeActionMapping();
      setStatus(`Action 后台任务已启动 · ${episodeIds.length} 个 Episode`); toast(`已提交 Action 生成 · ${episodeIds.length} 个 Episode`, ""); monitorBatchAnalysis(job.id);
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { button.disabled = false; }
  }
  function renderAnalysisThreadStatus() {
    const node = $("#analysisThreadStatus"), button = $("#cancelAnalysisButton");
    const active = [...state.analysisJobs.values()].filter(job => ["queued", "running", "cancelling"].includes(job.status || job.state));
    const current = active[active.length - 1], stopping = (current?.status || current?.state) === "cancelling";
    node.classList.toggle("running", Boolean(active.length));
    node.textContent = active.length ? `${active.length} 个后台任务 · ${active[active.length - 1].message || "运行中"}` : "后台线程空闲";
    button.classList.toggle("hidden", !current);
    button.disabled = !current || stopping;
    button.querySelector("span").textContent = stopping ? "正在终止" : "终止任务";
  }
  async function cancelActiveAnalysis() {
    const active = [...state.analysisJobs.values()].filter(job => ["queued", "running", "cancelling"].includes(job.status || job.state));
    const current = active[active.length - 1];
    if (!current || (current.status || current.state) === "cancelling") return;
    if (!window.confirm("确定终止当前后台任务？已经完成的 Episode 会保留。")) return;
    const button = $("#cancelAnalysisButton"); button.disabled = true;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(current.id)}/cancel`, { method: "POST" });
      state.analysisJobs.set(job.id, job); renderAnalysisThreadStatus();
      setStatus(job.message || "正在终止任务…");
    } catch (error) {
      toast(error.message, "error"); setStatus(error.message); renderAnalysisThreadStatus();
    }
  }
  async function restoreAnalysisJobs(datasetId) {
    try {
      const payloads = await Promise.all([
        api(`/api/datasets/${encodeURIComponent(datasetId)}/qwen-trim-jobs?active_only=true`).catch(() => ({ items: [] })),
        api(`/api/datasets/${encodeURIComponent(datasetId)}/curation-jobs?active_only=true`).catch(() => ({ items: [] })),
        api(`/api/datasets/${encodeURIComponent(datasetId)}/action-jobs?active_only=true`).catch(() => ({ items: [] })),
      ]);
      if (state.dataset?.id !== datasetId) return;
      for (const job of payloads.flatMap(payload => payload.items || [])) {
        state.analysisJobs.set(job.id, job);
        monitorBatchAnalysis(job.id);
      }
      renderAnalysisThreadStatus();
    } catch (_) { /* No recoverable jobs is normal. */ }
  }
  async function monitorBatchAnalysis(jobId) {
    if (state.analysisMonitors.has(jobId)) return;
    state.analysisMonitors.add(jobId);
    try {
      while (true) {
        const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
        state.analysisJobs.set(jobId, job); renderAnalysisThreadStatus(); setProgress(job.progress, job.message);
        const jobStatus = job.status || job.state;
        if (jobStatus === "complete" || jobStatus === "completed") {
          await loadChangeCatalog();
          const currentCurationItem = (job.result?.items || []).find(item => item.episode_id === state.episode?.id && item.status === "completed");
          if (["paper_curation", "full_pipeline"].includes(job.operation) && currentCurationItem) {
            await loadCurationReport(state.episode.id, state.media?.file_id);
            await loadBehaviorAnnotation();
          }
          if (job.operation === "vlm_behavior" && state.episode) await loadBehaviorAnnotation();
          if (job.operation === "pose_recovery" && state.episode) { await updateJointOverlayStatus(); await updateFrame(state.frame); }
          if (job.operation === "action_mapping" && state.episode && (job.result?.items || []).some(item => item.episode_id === state.episode.id && ["completed", "skipped"].includes(item.status))) await loadActionMappingResult(state.episode.id);
          if (job.operation === "no_action_trim" && state.episode && (job.result?.items || []).some(item => item.episode_id === state.episode.id && item.status === "completed")) { const payload = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/no-action-trim`); if (!payload.source_video?.file_id || payload.source_video.file_id === state.media?.file_id) { state.annotations = payload; renderAnnotations(payload); if (state.yoloOverlay) installYoloOverlayReport(payload); } }
          if ((job.kind === "qwen_action_trim" || job.result?.operation === "qwen_action_trim") && state.episode && (job.result?.items || []).some(item => item.episode_id === state.episode.id && item.status === "completed")) { const payload = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/qwen-action-trim`); if (!payload.source_video?.file_id || payload.source_video.file_id === state.media?.file_id) { state.annotations = payload; renderAnnotations(payload); } }
          const failures = Number(job.result?.failure_count || 0), total = Number(job.result?.episode_count || 0), skipped = Number(job.result?.skipped_count || 0);
          const skippedText = skipped ? ` · ${skipped} 个已复用并跳过` : "";
          const vlmText = ["paper_curation", "full_pipeline"].includes(job.operation)
            ? ` · VLM 请求 ${Number(job.result?.vlm_requested_count || 0)} · 复用 ${Number(job.result?.vlm_reused_count || 0)} · 跳过 ${Number(job.result?.vlm_skipped_count || 0)}`
            : "";
          const fullText = job.operation === "full_pipeline" ? `${vlmText} · ${Number(job.result?.pair_count || 0)} 对输出 · ${job.result?.output_root || "未生成目录"}` : vlmText;
          toast(failures ? `后台任务完成 · ${total - failures}/${total}${skippedText} · ${failures} 个失败${fullText}` : `后台任务完成 · ${total} 个 Episode${skippedText}${fullText}`, failures ? "error" : "");
          setStatus(job.operation === "full_pipeline" ? `${job.message}${fullText}` : `${job.message} · 结果已暂存到 .alicePD`); return;
        }
        if (jobStatus === "cancelled") {
          const completed = Number(job.completed_count || 0), total = Number(job.episode_count || 0);
          const summary = total ? `任务已终止 · 已完成 ${completed}/${total} Episodes` : "任务已终止";
          toast(summary, ""); setStatus(summary); return;
        }
        if (jobStatus === "failed") throw new Error(job.error || job.message || "后台任务失败");
        await new Promise(resolve => setTimeout(resolve, 700));
      }
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { state.analysisMonitors.delete(jobId); state.analysisJobs.delete(jobId); renderAnalysisThreadStatus(); if (!state.analysisJobs.size) setProgress(null); }
  }
  async function submitScopedAnalysis() {
    if (!state.dataset || !state.analysisOperation) return;
    if (!validateTrimSettings()) return;
    const all = $("input[name='analysisScope']:checked")?.value === "all";
    const episodeIds = all ? state.dataset.episodes.map(item => item.id) : [$("#analysisEpisodeSelect").value];
    const mediaFileIds = {};
    if (analysisNeedsMedia()) {
      const referenceEpisode = state.dataset.episodes.find(item => item.id === $("#analysisEpisodeSelect").value);
      const referenceMedia = (referenceEpisode?.media_streams || []).find(item => item.file_id === $("#analysisMediaSelect").value);
      if (!referenceMedia) { toast("请先指定一个可用视频流", "error"); return; }
      for (const episodeId of episodeIds) {
        const episode = state.dataset.episodes.find(item => item.id === episodeId);
        const streams = episode?.media_streams || [];
        let media = streams.find(item => item.file_id === referenceMedia.file_id);
        if (all) {
          if (streams.length === 1) media = streams[0];
          else {
            const select = $$(".analysis-media-map-select", $("#analysisMediaMapList")).find(item => item.dataset.episodeId === episodeId);
            media = streams.find(item => item.file_id === select?.value);
          }
        }
        if (!media) { toast(`${episode?.name || episodeId} 需要明确选择视频流`, "error"); return; }
        mediaFileIds[episodeId] = media.file_id;
      }
    }
    const button = $("#startScopedAnalysis"); button.disabled = true;
    try {
      const qwenTrim = state.analysisOperation === "qwen_trim", full = state.analysisOperation === "full_pipeline", curation = state.analysisOperation === "paper_curation";
      const endpoint = curation || full ? `/api/datasets/${encodeURIComponent(state.dataset.id)}/curation-jobs` : qwenTrim ? `/api/datasets/${encodeURIComponent(state.dataset.id)}/qwen-trim-jobs` : `/api/datasets/${encodeURIComponent(state.dataset.id)}/analysis-jobs`;
      const trimConfig = trimRequestConfig();
      const fullActionConfig = full ? fullActionRequestConfig() : {};
      if (fullActionConfig === null) return;
      const payload = curation || full
        ? { episode_ids: episodeIds, media_file_ids: mediaFileIds, ...curationRequestConfig(), ...fullActionConfig, full_pipeline: full, force_vlm: full && Boolean($("#forceAnalysis")?.checked) }
        : qwenTrim
        ? { episode_ids: episodeIds, all_episodes: all, media_file_ids: mediaFileIds, ...trimConfig }
        : { operation: state.analysisOperation, episode_ids: episodeIds, media_file_ids: mediaFileIds, sample_count: 18, sample_fps: 4, proximity_threshold: 0.04, max_gap_seconds: 0.5, min_valid_seconds: 0.3, force: state.analysisOperation === "vlm_behavior" && Boolean($("#forceAnalysis")?.checked), ...trimConfig };
      const job = await api(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      state.analysisJobs.set(job.id, job); renderAnalysisThreadStatus(); closeAnalysisScope();
      setStatus(`后台线程已启动 · ${episodeIds.length} 个 Episode`); toast(`已提交后台任务 · ${episodeIds.length} 个 Episode`, "");
      monitorBatchAnalysis(job.id);
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { button.disabled = false; }
  }
  async function waitBehaviorJob(id) {
    while (true) {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`); setProgress(job.progress, job.message);
      const status = job.status || job.state;
      if (status === "complete" || status === "completed") return job.result;
      if (status === "failed") throw new Error(job.error || job.message || "VLM 行为标注失败");
      await new Promise(resolve => setTimeout(resolve, 700));
    }
  }
  async function annotateBehavior(event) {
    openAnalysisScope("vlm_behavior", event?.currentTarget); return;
    if (!state.dataset || !await ensureEpisodeSelected()) return;
    const button = $("#behaviorAnnotateButton"); button.disabled = true; setProgress(2, "准备 VLM 行为标注");
    try {
      const job = await api(`/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/annotate-behavior`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample_count: 18 }) });
      const result = await waitBehaviorJob(job.id); renderBehaviorAnnotation(result); await loadChangeCatalog();
      toast(`行为标注完成，已暂存待应用 · ${result.task_label || "other"} · 目标 ${(result.primary_targets || []).length} 个`, "");
      setStatus(`VLM 行为标注已写入 .alicePD · ${result.artifacts?.behavior || ""}`);
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { button.disabled = !state.episode; setProgress(null); }
  }
  async function recoverInitialPose(event) {
    openAnalysisScope("pose_recovery", event?.currentTarget); return;
    if (!state.dataset || !await ensureEpisodeSelected()) return;
    const button = $("#poseRecoveryButton"); button.disabled = true; setProgress(12, "检查 mocap 缺帧与腕部视频特征");
    try {
      const base = `/api/datasets/${encodeURIComponent(state.dataset.id)}/episodes/${encodeURIComponent(state.episode.id)}/pose-recovery`;
      const status = await api(`${base}/status`);
      if (!status.available) throw new Error("当前 Episode 没有可恢复的手部 mocap 数据");
      if (!status.source_gap_exists) { toast("当前 Episode 的手部初始位姿完整", ""); return; }
      if (!status.needed && status.artifact_exists) { toast("当前 Episode 已有位姿恢复结果", ""); return; }
      setProgress(48, "运行 ORB / RANSAC 视觉里程计校验");
      const result = await api(base, { method: "POST" });
      setProgress(88, "写入 .alicePD 位姿恢复旁路文件");
      await loadChangeCatalog(); await updateJointOverlayStatus(); await updateFrame(state.frame);
      const count = Number(result.recovered_frame_count || 0);
      toast(count ? `已恢复 ${count} 个手部初始位姿帧` : "当前 Episode 不需要位姿补全", "");
      setStatus(`SLAM/VO 位姿恢复完成 · ${count} 帧 · 源 H5 保持只读`);
    } catch (error) { toast(error.message, "error"); setStatus(error.message); }
    finally { button.disabled = !state.episode; setProgress(null); }
  }
  function bind() {
    ensureCurationQualityGapControl();
    ensureS1RepairControls();
    ensureFullActionControls();
    $$("[data-ribbon]").forEach(button => button.addEventListener("click", () => { $$("[data-ribbon]").forEach(item => item.classList.toggle("active", item === button)); $$("[data-pane]").forEach(pane => pane.classList.toggle("active", pane.dataset.pane === button.dataset.ribbon)); }));
    $$("[data-view-target]").forEach(button => button.addEventListener("click", () => showView(button.dataset.viewTarget)));
    $$("[data-inspector]").forEach(button => button.addEventListener("click", () => activateInspector(button.dataset.inspector)));
    $("#openFolderButton").addEventListener("click", openFolder); $("#datasetSelect").addEventListener("change", event => loadCollectionDataset(event.target.value)); $("#refreshButton").addEventListener("click", refreshDataset); $("#analyzeSchemaButton").addEventListener("click", understandSchema); $("#excludeFileButton").addEventListener("click", openExcludeFileModal); $("#confirmExcludeFile").addEventListener("click", confirmExcludeFile); $("#manualRangeButton").addEventListener("click", openManualRange); $("#curationPipelineButton").addEventListener("click", event => openAnalysisScope("paper_curation", event.currentTarget)); $("#fullPipelineButton").addEventListener("click", event => openAnalysisScope("full_pipeline", event.currentTarget)); $("#actionMappingButton").addEventListener("click", event => openActionMapping(event.currentTarget)); $("#videoSmoothButton").addEventListener("click", event => openAnalysisScope("video_smoothing", event.currentTarget)); $("#poseRecoveryButton").addEventListener("click", recoverInitialPose); $("#behaviorAnnotateButton").addEventListener("click", annotateBehavior); $("#noActionTrimButton").addEventListener("click", event => openAnalysisScope("no_action_trim", event.currentTarget)); $("#qwenTrimButton").addEventListener("click", event => openAnalysisScope("qwen_trim", event.currentTarget)); $("#cancelAnalysisButton").addEventListener("click", cancelActiveAnalysis); $("#exportFolderButton").addEventListener("click", exportFolder); $("#downloadExport").addEventListener("click", downloadZip); $("#reviewChangesButton").addEventListener("click", openChangeModal); $("#applyChangesButton").addEventListener("click", openChangeModal); $("#confirmApplyChanges").addEventListener("click", applySelectedChanges); $("#changeConfirm").addEventListener("change", updateChangeApplyState); $("#modelButton").addEventListener("click", configureModal); $("#saveModel").addEventListener("click", saveModel);
    $$(".modal-close").forEach(button => button.addEventListener("click", () => $("#modelModal").classList.add("hidden"))); $$(".change-modal-close").forEach(button => button.addEventListener("click", closeChangeModal)); $$(".analysis-scope-close").forEach(button => button.addEventListener("click", closeAnalysisScope)); $$(".action-mapping-close").forEach(button => button.addEventListener("click", closeActionMapping)); $$(".manual-range-close").forEach(button => button.addEventListener("click", closeManualRange)); $$(".exclude-file-close").forEach(button => button.addEventListener("click", closeExcludeFileModal)); $$("input[name='manualRangeState']").forEach(input => input.addEventListener("change", updateManualRangeSummary)); $("#manualRangeStart").addEventListener("input", updateManualRangeSummary); $("#manualRangeEnd").addEventListener("input", updateManualRangeSummary); $("#manualStartCurrent").addEventListener("click", () => setManualRangeCurrent("#manualRangeStart")); $("#manualEndCurrent").addEventListener("click", () => setManualRangeCurrent("#manualRangeEnd")); $("#saveManualRange").addEventListener("click", saveManualRange); $$("input[name='analysisScope']").forEach(input => input.addEventListener("change", updateAnalysisScope)); $$("input[name='actionScope']").forEach(input => input.addEventListener("change", updateActionMappingScope)); $("#actionRobotProfile").addEventListener("change", renderActionProfileNote); $("#fullOutputFormat").addEventListener("change", updateAnalysisScope); $("#fullGenerateAction").addEventListener("change", renderFullActionSettings); $("#fullRobotProfile").addEventListener("change", renderFullActionSettings); $("#startActionMapping").addEventListener("click", submitActionMapping); $("#analysisEpisodeSelect").addEventListener("change", updateAnalysisMediaOptions); $("#analysisMediaSelect").addEventListener("change", () => { renderAnalysisMediaMap(); updateCurationPreflight(); }); $("#fillAnalysisMediaByName").addEventListener("click", fillAnalysisMediaByName); $("#startScopedAnalysis").addEventListener("click", submitScopedAnalysis); $("#clearCurationStageFilter").addEventListener("click", () => { state.curationStageFilter = null; $$(".curation-stage-row", $("#curationStageList")).forEach(item => item.classList.remove("active")); renderCurationTrack(); }); $("#modelType").addEventListener("change", () => { const qwen = $("#modelType").value === "qwen"; $("#localModelFields").classList.toggle("hidden", qwen); $("#qwenFields").classList.toggle("hidden", !qwen); });
    $("#treeMode").addEventListener("change", event => { state.treeMode = event.target.value; state.treeGroups = new Map(); state.treeExpanded = new Set(); state.treeCollapsed = new Set(); if (state.treeSelection?.kind === "episode") state.treeSelection = null; renderResolvedTree(); updateTreeSelectionUI(); }); $("#treeSearch").addEventListener("input", () => { clearTimeout(state.treeSearchTimer); state.treeSearchTimer = setTimeout(renderResolvedTree, 120); }); $("#expandButton").addEventListener("click", () => { state.treeAutoCollapse = false; state.treeCollapsed = new Set(); state.treeExpanded = new Set(); renderResolvedTree(); }); $("#collapseButton").addEventListener("click", () => { state.treeAutoCollapse = true; state.treeExpanded = new Set(); state.treeCollapsed = new Set(); renderResolvedTree(); }); $("#themeButton").addEventListener("click", () => document.body.classList.toggle("dark")); $("#detailButton").addEventListener("click", () => $("#inspector").classList.toggle("closed"));
    $("#frameSlider").addEventListener("input", event => updateFrame(event.target.value)); $("#prevFrame").addEventListener("click", () => updateFrame(state.frame - 1)); $("#nextFrame").addEventListener("click", () => updateFrame(state.frame + 1)); $("#transportPlay").addEventListener("click", togglePlay); $("#playButton").addEventListener("click", togglePlay); $("#jointOverlayButton").addEventListener("click", toggleJointOverlay); $("#jointIndexButton").addEventListener("click", toggleJointIndices); $("#yoloOverlayButton").addEventListener("click", toggleYoloOverlay); $$(".segmented button").forEach(button => button.addEventListener("click", () => { $$(".segmented button").forEach(item => item.classList.toggle("active", item === button)); updateFrame(state.frame); })); $("#zoomIn").addEventListener("click", () => zoom(0.1)); $("#zoomOut").addEventListener("click", () => zoom(-0.1));
    $("#behaviorRemoveSelect").addEventListener("change", updateBehaviorRemovalState); $("#behaviorRemoveButton").addEventListener("click", removeBehaviorPhase);
    $("#frameDataFollow").addEventListener("change", event => { state.frameData.follow = event.target.checked; if (state.frameData.follow && state.episode) scheduleFrameData(state.frame, true); });
    $("#frameDataField").addEventListener("change", event => { state.frameData.field = event.target.value; if (["hdf5", "numpy", "parquet"].includes(state.frameData.mode)) openFilePreview(state.file, null, state.frameData.field); else loadFrameData(state.frameData.index, state.frameData.field); });
    $("#frameDataPrev").addEventListener("click", () => loadFrameData(Math.max(0, state.frameData.index - 1), state.frameData.field)); $("#frameDataNext").addEventListener("click", () => loadFrameData(Math.min(state.frameData.count - 1, state.frameData.index + 1), state.frameData.field));
    $("#frameDataIndex").addEventListener("change", event => loadFrameData(event.target.value, state.frameData.field)); $("#frameDataSlider").addEventListener("input", event => scheduleFrameData(event.target.value, true));
    $("#videoPlayer").addEventListener("seeked", () => { if (state.nativePreview) updateNativeVideoFrame(); }); window.addEventListener("resize", () => { if (state.nativePreview && state.jointOverlay) requestJointGeometry(state.frame, true); if (state.yoloOverlay) renderYoloOverlayFrame(state.frame); });
    ["yoloProximityThreshold", "yoloMaxGapSeconds", "yoloMinValidSeconds", "qwenTrimConfidence", "qwenTrimMaxGapSeconds", "qwenTrimMinValidSeconds"].forEach(id => $("#" + id).addEventListener("input", updateTrimSettingOutputs));
    document.addEventListener("keydown", event => { if (event.key === "Escape" && !$("#analysisScopeModal").classList.contains("hidden")) closeAnalysisScope(); });
  }
  function toggleJointOverlay() {
    if (!state.jointOverlayAvailable) { const message = "当前 Episode 未发现可投影的 joint/transform 数据，请切换到带 mocap 或 joint state 的 Episode"; toast(message, "error"); setStatus(message); return; }
    state.jointOverlay = !state.jointOverlay;
    const button = $("#jointOverlayButton"); button.classList.toggle("active", state.jointOverlay); button.setAttribute("aria-pressed", String(state.jointOverlay));
    if (!state.jointOverlay) {
      state.jointGeometryPendingFrame = null; clearTimeout(state.jointGeometryTimer); state.jointGeometryTimer = null;
      if (state.jointGeometryAbortController) state.jointGeometryAbortController.abort();
      resetJointIndices(); $("#jointOverlayCanvas").classList.add("hidden"); $("#jointOverlayHint").classList.add("hidden"); return;
    }
    $("#jointIndexButton").disabled = false;
    if (state.nativePreview) requestJointGeometry(state.frame, true); else updateFrame(state.frame);
  }
  function toggleJointIndices() {
    const button = $("#jointIndexButton"); if (button.disabled || !state.jointOverlay) return;
    state.jointIndices = !state.jointIndices; button.classList.toggle("active", state.jointIndices); button.setAttribute("aria-pressed", String(state.jointIndices));
    if (state.nativePreview) {
      if (state.jointGeometryCurrent) drawJointGeometry(state.jointGeometryCurrent); else requestJointGeometry(state.frame, true);
    } else updateFrame(state.frame);
  }
  function zoom(delta) { state.zoom = Math.max(.5, Math.min(2, state.zoom + delta)); $("#frameImage").style.transform = `scale(${state.zoom})`; $("#videoPlayer").style.transform = `scale(${state.zoom})`; $("#yoloOverlayCanvas").style.transform = `scale(${state.zoom})`; $("#jointOverlayCanvas").style.transform = `scale(${state.zoom})`; $("#zoomValue").textContent = `${Math.round(state.zoom * 100)}%`; }
  function togglePlay() {
    if (!state.episode) return;
    if (state.playing) { stopPlayback(); return; }
    if (state.nativePreview?.status === "ready") { const video = $("#videoPlayer"); state.playing = true; state.playbackToken += 1; video.muted = true; video.play().then(() => scheduleNativeFrameCallback()).catch(error => { state.playing = false; setPreviewStatus(`播放失败 · ${error.name || "Error"}`, "failed"); console.warn(`Native video play failed: ${error.name || "Error"}: ${error.message || ""}`); }); $("#transportPlay").innerHTML = '<i data-lucide="pause"></i>'; $("#playButton").innerHTML = $("#transportPlay").innerHTML; lucide.createIcons(); return; }
    const media = state.media || state.episode;
    state.playing = true; state.playbackToken += 1; state.playbackStartedAt = performance.now(); state.playbackStartFrame = state.frame;
    $("#transportPlay").innerHTML = '<i data-lucide="pause"></i>'; $("#playButton").innerHTML = $("#transportPlay").innerHTML; lucide.createIcons(); tick(state.playbackToken, media);
  }
  async function tick(token, media) {
    if (!state.playing || token !== state.playbackToken || !media?.frame_count) return;
    const elapsedFrames = Math.max(1, Math.floor((performance.now() - state.playbackStartedAt) * Number(media.fps || 30) / 1000));
    const target = (state.playbackStartFrame + elapsedFrames) % media.frame_count;
    const loaded = await updateFrame(target === state.frame ? (state.frame + 1) % media.frame_count : target, true);
    if (!state.playing || token !== state.playbackToken) return;
    state.timer = setTimeout(() => tick(token, media), loaded ? Math.max(20, Math.min(50, 1000 / Number(media.fps || 30))) : 180);
  }
  async function boot() { lucide.createIcons(); bind(); try { const [health, datasets] = await Promise.all([api("/api/health"), api("/api/datasets")]); state.models = health.models || null; $("#serviceDot").classList.add("ready"); $("#serviceStatus").textContent = "后端已连接"; renderModels(); if (datasets.items?.length) { const latest = await api(`/api/datasets/${encodeURIComponent(datasets.items[0].id)}`); renderDataset(latest); } if (new URLSearchParams(location.search).get("launch") === "open-folder") setTimeout(openFolder, 180); } catch (error) { $("#serviceDot").classList.add("error"); $("#serviceStatus").textContent = "后端不可用"; setStatus(error.message); } }
  boot();
})();
