const state = {
  dashboard: null,
  selected: new Set(),
  jobs: [],
  detail: null,
  videoId: null,
  chapterIndex: 0,
  markdown: "",
  markdownDirty: false,
  glossaryCandidates: [],
  pendingGlossaryTerms: [],
  glossaryHits: [],
  glossaryApplyScope: null,
  activeJobId: null,
  notifiedInterruptedJobIds: new Set(),
  videoResumeTime: null,
};

const $ = selector => document.querySelector(selector);

function node(tag, options = {}, children = []) {
  const element = document.createElement(tag);
  if (options.className) element.className = options.className;
  if (options.text !== undefined) element.textContent = options.text;
  if (options.title) element.title = options.title;
  for (const child of children) if (child) element.append(child);
  return element;
}

function formatTime(ms = 0) {
  const total = Math.max(0, Math.floor(Number(ms || 0) / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function displayVideoTitle(value = "") {
  return String(value)
    .replace(/__[0-9a-f]{16}(?=\.[^.]+$|$)/i, "")
    .replace(/\.(mp4|mov|mkv|avi|webm|m4v)$/i, "");
}

function durationText(start, end) {
  return formatTime(Math.max(0, Number(end || 0) - Number(start || 0)));
}

function paragraphText(value = "") {
  const parts = String(value).replace(/\r/g, "").split("\n")
    .map(line => line.replace(/\s+/g, " ").trim()).filter(Boolean);
  return parts.reduce((result, part) => {
    const needsSpace = result && /[A-Za-z0-9,.;:!?)]$/.test(result) && /^[A-Za-z0-9(]/.test(part);
    return `${result}${needsSpace ? " " : ""}${part}`;
  }, "");
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const detail = (await response.json()).detail;
      message = Array.isArray(detail) ? detail.map(item => item.msg || String(item)).join("；") : detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

let toastTimer;
function toast(message, error = false) {
  const root = $("#toast");
  root.textContent = message;
  root.className = `toast${error ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => root.classList.add("hidden"), 3200);
}
let currentCloudStatus = null;

function renderCloudStatus(status) {
  currentCloudStatus = status;
  const button = $("#apiConnectionButton");
  button.classList.toggle("api-connected", Boolean(status.connected));
  const statusText = status.connected
    ? `已连接 · ${status.model} · ${status.key_hint}${status.persisted ? " · 已随文件夹保存" : " · 当前来自本机环境变量"}`
    : "尚未连接 API";
  $("#apiConnectionStatus").textContent = statusText;
  $("#apiConnectionStatus").classList.toggle("connected", Boolean(status.connected));
  $("#apiModelInput").value = status.model || "glm-5.3";
  $("#apiUrlInput").value = status.api_url || "https://open.bigmodel.cn/api/paas/v4/chat/completions";
  $("#apiKeyInput").value = "";
  $("#apiKeyInput").placeholder = status.connected ? "留空则继续使用已保存密钥" : "请输入 API Key";
  $("#disconnectApiButton").disabled = !status.connected;
}

async function refreshCloudStatus() {
  const status = await api("/api/cloud/status");
  renderCloudStatus(status);
  return status;
}

async function openApiDialog() {
  $("#apiDialog").showModal();
  $("#apiConnectionStatus").textContent = "正在读取连接状态…";
  try { await refreshCloudStatus(); }
  catch (error) { $("#apiConnectionStatus").textContent = error.message; }
}

async function submitApiConnection(event) {
  event.preventDefault();
  const button = $("#connectApiButton");
  button.disabled = true;
  button.textContent = "正在测试…";
  $("#apiConnectionStatus").textContent = "正在测试 API 和模型，请稍候…";
  try {
    const status = await api("/api/cloud/connect", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        api_key: $("#apiKeyInput").value.trim() || null,
        model: $("#apiModelInput").value.trim(),
        api_url: $("#apiUrlInput").value.trim(),
      }),
    });
    renderCloudStatus(status);
    toast(`API连接成功，当前模型：${status.model}`);
    $("#apiDialog").close();
  } catch (error) {
    $("#apiConnectionStatus").textContent = `连接失败：${error.message}`;
    $("#apiConnectionStatus").classList.remove("connected");
  } finally {
    button.disabled = false;
    button.textContent = "测试并保存";
  }
}

async function disconnectApiConnection() {
  if (!confirm("确定断开 API？断开后，一键处理和 AI 生成功能将不可用。")) return;
  try {
    const status = await api("/api/cloud/connect", {method: "DELETE"});
    renderCloudStatus(status);
    toast("API已断开");
  } catch (error) { toast(`断开失败：${error.message}`, true); }
}

function pauseDetailVideo() {
  const video = $("#detailVideo");
  if (video && !video.paused) video.pause();
}

function pauseEditorVideo() {
  const video = $("#editorVideo");
  if (video && !video.paused) video.pause();
}

function showView(id) {
  if (id !== "videoView") pauseDetailVideo();
  if (id !== "editorView") pauseEditorVideo();
  for (const view of ["dashboardView", "videoView", "editorView"]) {
    $(`#${view}`).classList.toggle("hidden", view !== id);
  }
  window.scrollTo({top: 0});
}

function navigate(path) {
  history.pushState({}, "", path);
  route().catch(error => toast(error.message, true));
}

function visibleRecords() {
  const query = $("#searchInput").value.trim().toLowerCase();
  return (state.dashboard?.records || []).filter(row => !query || displayVideoTitle(row.title).toLowerCase().includes(query));
}

async function loadDashboard() {
  state.dashboard = await api("/api/dashboard");
  const ids = new Set(state.dashboard.records.map(row => row.id));
  state.selected = new Set([...state.selected].filter(id => ids.has(id)));
  renderSummary();
  renderRows();
}

function renderSummary() {
  const counts = state.dashboard?.counts || {};
  const cards = [
    ["视频数", counts.videos || 0],
    ["待审核数", counts.pending_review || 0],
    ["已处理完成数", counts.completed || 0],
  ];
  $("#summaryCards").replaceChildren(...cards.map(([label, value]) =>
    node("article", {className: "summary-card"}, [
      node("div", {className: "value", text: value}),
      node("div", {className: "label", text: label}),
    ])
  ));
}

function badge(text, type) {
  return node("span", {className: `badge ${type}`, text});
}

function reviewStatus(row) {
  if (!row.has_segments) return badge("未生成", "missing");
  return row.review_status === "confirmed" ? badge("已入库", "ok") : badge("待审核", "pending");
}

function renderRows() {
  const rows = visibleRecords();
  const root = $("#videoRows");
  root.replaceChildren();
  for (const row of rows) {
    const displayTitle = displayVideoTitle(row.title);
    const checkbox = node("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(row.id);
    checkbox.setAttribute("aria-label", `选择 ${displayTitle}`);
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.selected.add(row.id) : state.selected.delete(row.id);
      updateSelection();
    });

    const edit = node("button", {className: "text-button", text: "编辑"});
    edit.addEventListener("click", () => navigate(`/videos/${row.id}`));
    const remove = node("button", {className: "text-button delete", text: "删除"});
    remove.addEventListener("click", () => removeVideo(row));

    root.append(node("tr", {}, [
      node("td", {className: "check-cell"}, [checkbox]),
      node("td", {className: "video-name"}, [node("strong", {text: displayTitle})]),
      node("td", {text: row.duration_ms ? formatTime(row.duration_ms) : "—"}),
      node("td", {}, [row.has_raw ? badge("已生成", "ok") : badge("未生成", "missing")]),
      node("td", {}, [reviewStatus(row)]),
      node("td", {}, [row.has_segments ? badge(`${row.segment_count} 章`, "ok") : badge("未生成", "missing")]),
      node("td", {}, [node("div", {className: "row-actions"}, [edit, remove])]),
    ]));
  }
  $("#emptyState").classList.toggle("hidden", rows.length > 0);
  const selectedVisible = rows.filter(row => state.selected.has(row.id)).length;
  $("#selectAll").checked = rows.length > 0 && selectedVisible === rows.length;
  $("#selectAll").indeterminate = selectedVisible > 0 && selectedVisible < rows.length;
  updateSelection();
}

function updateSelection() {
  $("#selectionCount").textContent = `已选择 ${state.selected.size} 个`;
}

async function uploadFiles() {
  const files = [...$("#fileInput").files];
  if (!files.length) return toast("请先选择视频文件", true);
  const button = $("#uploadButton");
  const data = new FormData();
  files.forEach(file => data.append("files", file));
  button.disabled = true;
  $("#uploadState").textContent = `正在上传 ${files.length} 个文件…`;
  try {
    await api("/api/uploads", {method: "POST", body: data});
    $("#fileInput").value = "";
    $("#uploadState").textContent = "上传完成";
    toast("视频上传完成，可以勾选后处理");
    await loadDashboard();
  } catch (error) {
    $("#uploadState").textContent = "上传失败";
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function startProcessing() {
  if (!state.selected.size) return toast("请先勾选需要处理的视频", true);
  const button = $("#processButton");
  button.disabled = true;
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({stage: "pipeline", video_ids: [...state.selected]}),
    });
    state.selected.clear();
    state.activeJobId = job.id;
    renderRows();
    toast("已开始一键处理，完成后会进入待审核");
    await loadJobs();
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function removeVideo(row) {
  if (!confirm(`确定删除“${displayVideoTitle(row.title)}”及其处理数据吗？此操作无法撤销。`)) return;
  try {
    await api(`/api/videos/${row.id}`, {method: "DELETE"});
    state.selected.delete(row.id);
    toast("视频及相关处理数据已删除");
    await loadDashboard();
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadJobs() {
  try {
    const previousActiveJobId = state.activeJobId;
    state.jobs = (await api("/api/jobs")).jobs;
    const newlyInterrupted = state.jobs.filter(job =>
      job.status === "interrupted" && !state.notifiedInterruptedJobIds.has(job.id)
    );
    for (const job of newlyInterrupted) state.notifiedInterruptedJobIds.add(job.id);
    if (newlyInterrupted.length) {
      toast("上次任务因服务中断未完成，可重新选择视频继续处理", true);
    }
    const hasActiveJob = state.jobs.some(job =>
      ["running", "queued", "cancelling"].includes(job.status)
    );
    const activeJobJustFinished = Boolean(previousActiveJobId) && !state.jobs.some(job =>
      job.id === previousActiveJobId && ["running", "queued", "cancelling"].includes(job.status)
    );
    renderJobs();
    if (hasActiveJob || activeJobJustFinished) await loadDashboard();
  } catch (_) {}
}

function renderJobs() {
  const panel = $("#taskPanel");
  const active = state.jobs.find(job => ["queued", "running", "cancelling"].includes(job.status));
  if (!active) {
    if (state.activeJobId) {
      const finished = state.jobs.find(job => job.id === state.activeJobId);
      if (finished?.status === "completed") toast("视频处理已完成，现已进入待审核");
      else if (finished?.status === "cancelled") toast("任务已终止，已完成的阶段会保留");
      else if (finished?.status === "failed") toast("任务处理失败，请查看终端日志", true);
    }
    state.activeJobId = null;
    panel.classList.add("hidden");
    return;
  }
  state.activeJobId = active.id;
  const runningItem = active.items.find(item => item.status === "running")
    || active.items.find(item => item.status === "queued") || active.items[0];
  const logStages = [...String(runningItem?.log || "").matchAll(/===\s*(asr|cleanup|segments)\s*===/g)];
  const stage = runningItem?.current_stage || active.current_stage
    || (logStages.length ? logStages[logStages.length - 1][1] : null);
  const stageOrder = ["asr", "cleanup", "segments"];
  const stageLabels = {asr: "语音识别", cleanup: "文本清洗", segments: "章节生成"};
  const stageIndex = stage ? stageOrder.indexOf(stage) : -1;
  const fallbackProgress = stage === "asr" ? 8 : stage === "cleanup" ? 74 : stage === "segments" ? 88 : 2;
  const progressRanges = {asr: [5, 70], cleanup: [72, 85], segments: [87, 100]};
  let logProgress = null;
  for (const match of String(runningItem?.log || "").matchAll(/^PIPELINE_PROGRESS (.+)$/gm)) {
    try {
      const event = JSON.parse(match[1]);
      const total = Number(event.total);
      if (event.stage === stage && total > 0 && progressRanges[stage]) {
        const fraction = Math.max(0, Math.min(1, Number(event.completed) / total));
        const [start, end] = progressRanges[stage];
        logProgress = {
          value: start + Math.round((end - start) * fraction),
          detail: String(event.detail || "").trim(),
        };
      }
    } catch (_) {}
  }
  const progress = Math.max(0, Math.min(100,
    Math.max(Number(active.progress ?? fallbackProgress), Number(logProgress?.value ?? 0))));
  const progressDetail = String(
    runningItem?.progress_detail || active.progress_detail || logProgress?.detail || ""
  ).trim();
  const elapsedMs = active.started_at ? Math.max(0, Date.now() - Date.parse(active.started_at)) : 0;
  const record = state.dashboard?.records.find(row => row.id === runningItem?.video_id);
  const itemIndex = Math.max(0, active.items.indexOf(runningItem));
  $("#taskVideoTitle").textContent = displayVideoTitle(record?.title || "视频处理任务");
  $("#taskStageText").textContent = active.status === "cancelling"
    ? "正在终止任务…"
    : active.status === "queued"
      ? "等待开始"
      : stage
        ? `第 ${stageIndex + 1}/3 阶段 · ${stageLabels[stage]}${progressDetail ? ` · ${progressDetail}` : "中"}`
        : "正在准备处理";
  $("#taskProgressText").textContent = active.started_at
    ? `${progress}% · 已运行 ${formatTime(elapsedMs)}` : `${progress}%`;
  const fill = $("#taskProgressFill");
  fill.style.width = `${Math.max(progress, 3)}%`;
  fill.classList.toggle("running", active.status === "running");
  fill.parentElement.setAttribute("aria-valuenow", String(progress));
  $("#taskStageSteps").replaceChildren(...stageOrder.map((value, index) => node("span", {
    className: `task-stage-step${index < stageIndex ? " done" : index === stageIndex ? " active" : ""}`,
    text: stageLabels[value],
  })));
  $("#taskQueueText").textContent = active.items.length > 1
    ? `第 ${itemIndex + 1}/${active.items.length} 个视频；其余视频将依次处理。`
    : progressDetail || (stage === "asr" ? "语音识别通常耗时最长，进度条流动表示任务仍在运行。" : "任务正在运行，请勿拔出存储设备。");
  const cancelButton = $("#cancelTaskButton");
  cancelButton.disabled = active.status === "cancelling" || active.can_cancel !== true;
  cancelButton.textContent = active.status === "cancelling"
    ? "正在终止…" : active.can_cancel === true ? "终止任务" : "本次暂不可终止";
  panel.classList.remove("hidden");
}

async function cancelActiveJob() {
  const job = state.jobs.find(item => item.id === state.activeJobId);
  if (!job || !["queued", "running", "cancelling"].includes(job.status)) return;
  if (job.can_cancel !== true) {
    return toast("当前任务由更新前的服务启动；为避免损坏正在生成的数据，本次任务不会被强制中断。", true);
  }
  if (!confirm("确定终止当前任务吗？已经完成的处理阶段会保留，下次可继续处理。")) return;
  const button = $("#cancelTaskButton");
  button.disabled = true;
  button.textContent = "正在终止…";
  try {
    await api(`/api/jobs/${job.id}/cancel`, {method: "POST"});
    toast("已发送终止请求");
    await loadJobs();
  } catch (error) {
    toast(`终止失败：${error.message}`, true);
    button.disabled = false;
    button.textContent = "终止任务";
  }
}

function chapterRows() {
  return state.detail?.artifacts?.segments?.segments || [];
}

function currentChapter() {
  return chapterRows()[state.chapterIndex] || null;
}

function setChapter(index, seek = true) {
  const rows = chapterRows();
  if (!rows.length) return;
  state.chapterIndex = Math.max(0, Math.min(index, rows.length - 1));
  if (seek) {
    const video = $("#detailVideo");
    video.currentTime = Number(rows[state.chapterIndex].start_ms || 0) / 1000;
    video.play().catch(() => {});
  }
  renderChapterSelection();
}

function renderChapterSelection() {
  const chapter = currentChapter();
  if (!chapter) return;
  $("#currentChapterTitle").textContent = `第 ${state.chapterIndex + 1} 章 · ${chapter.title}`;
  $("#documentTitle").textContent = chapter.title;
  document.querySelectorAll(".chapter-card").forEach((card, index) => card.classList.toggle("active", index === state.chapterIndex));
  document.querySelectorAll(".chapter-dot").forEach((dot, index) => dot.classList.toggle("active", index === state.chapterIndex));
  renderChapterDocument(chapter);
}

function renderChapterDocument(chapter) {
  const root = $("#chapterDocument");
  const meta = node("blockquote", {}, [
    node("div", {text: `时间：${formatTime(chapter.start_ms)}–${formatTime(chapter.end_ms)} · 时长 ${durationText(chapter.start_ms, chapter.end_ms)}`}),
  ]);
  const keywords = node("div", {className: "keyword-row"});
  for (const item of chapter.keywords || []) keywords.append(node("span", {className: "keyword", text: item}));
  root.replaceChildren(
    meta,
    keywords,
    node("h3", {text: "章节摘要"}),
    node("p", {text: paragraphText(chapter.summary) || "（暂无摘要）"}),
    node("h3", {text: "章节正文"}),
    node("p", {text: paragraphText(chapter.content) || "（暂无正文）"}),
  );
}

function positionChapterDots(totalMs) {
  const total = Math.max(1, Number(totalMs || 1));
  document.querySelectorAll(".chapter-dot").forEach((dot, index) => {
    const chapter = chapterRows()[index];
    const percent = Number(chapter?.start_ms || 0) / total * 100;
    dot.style.left = `${Math.min(100, Math.max(0, percent))}%`;
  });
}

function syncPlayerControls() {
  const video = $("#detailVideo");
  const seek = $("#videoSeek");
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const percent = duration > 0 ? Math.min(100, Math.max(0, video.currentTime / duration * 100)) : 0;
  seek.value = String(Math.round(percent * 10));
  seek.style.setProperty("--played", `${percent}%`);
  $("#videoTime").textContent = `${formatTime(video.currentTime * 1000)} / ${formatTime(duration * 1000)}`;
  $("#playButton").textContent = video.paused ? "▶" : "❚❚";
  $("#playButton").setAttribute("aria-label", video.paused ? "播放" : "暂停");
}

function togglePlayback() {
  const video = $("#detailVideo");
  if (video.paused) video.play().catch(() => {});
  else video.pause();
}

function editorChapterBounds() {
  const chapter = currentChapter();
  return {
    start: Number(chapter?.start_ms || 0) / 1000,
    end: Number(chapter?.end_ms || 0) / 1000,
  };
}

function syncEditorPlayerControls() {
  const video = $("#editorVideo");
  const seek = $("#editorVideoSeek");
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const percent = duration > 0 ? Math.min(100, Math.max(0, video.currentTime / duration * 100)) : 0;
  seek.value = String(Math.round(percent * 10));
  seek.style.setProperty("--played", `${percent}%`);
  $("#editorVideoTime").textContent = `${formatTime(video.currentTime * 1000)} / ${formatTime(duration * 1000)}`;
  $("#editorPlayButton").textContent = video.paused ? "▶" : "❚❚";
  $("#editorPlayButton").setAttribute("aria-label", video.paused ? "播放" : "暂停");
}

function seekEditorVideo(seconds, play = false) {
  const video = $("#editorVideo");
  if (!Number.isFinite(seconds)) return;
  const duration = Number.isFinite(video.duration) ? video.duration : seconds;
  video.currentTime = Math.min(Math.max(0, seconds), Math.max(0, duration));
  syncEditorPlayerControls();
  if (play) video.play().catch(() => {});
}

function seekEditorToSentence(sentenceId) {
  const sentences = state.detail?.artifacts?.cleaned?.sentences || [];
  const sentence = sentences.find(item => Number(item.sentence_id ?? item.id) === Number(sentenceId));
  const fallback = Number(currentChapter()?.start_ms || 0);
  seekEditorVideo(Number(sentence?.start_ms ?? fallback) / 1000, true);
}

function openEditorVideoAtSentence(sentenceId) {
  $("#editorVideoPanel").classList.remove("hidden");
  const video = $("#editorVideo");
  if (video.readyState >= 1) seekEditorToSentence(sentenceId);
  else video.addEventListener("loadedmetadata", () => seekEditorToSentence(sentenceId), {once: true});
}

function openEditorVideoAtChapterStart() {
  $("#editorVideoPanel").classList.remove("hidden");
  const video = $("#editorVideo");
  const start = editorChapterBounds().start;
  if (video.readyState >= 1) seekEditorVideo(start, true);
  else video.addEventListener("loadedmetadata", () => seekEditorVideo(start, true), {once: true});
}

function closeEditorVideo() {
  pauseEditorVideo();
  $("#editorVideoPanel").classList.add("hidden");
}

function toggleEditorPlayback() {
  const video = $("#editorVideo");
  if (video.paused) video.play().catch(() => {});
  else video.pause();
}

function configureEditorVideo() {
  const chapter = currentChapter();
  if (!chapter) return;
  const video = $("#editorVideo");
  const {start, end} = editorChapterBounds();
  $("#editorChapterRange").textContent = `本章 ${formatTime(chapter.start_ms)}–${formatTime(chapter.end_ms)}`;
  $("#editorVideoPanel").classList.add("hidden");
  const hasRequested = state.videoResumeTime !== null && Number.isFinite(Number(state.videoResumeTime));
  const requested = Number(state.videoResumeTime);
  const target = hasRequested && requested >= start && requested <= end ? requested : start;
  const restorePosition = () => {
    seekEditorVideo(target, false);
    state.videoResumeTime = null;
  };
  video.src = `/api/videos/${state.videoId}/media`;
  if (video.readyState >= 1) restorePosition();
  else video.addEventListener("loadedmetadata", restorePosition, {once: true});
  syncEditorPlayerControls();
}

function renderVideo() {
  const document = state.detail?.artifacts?.segments;
  const record = state.dashboard?.records.find(row => row.id === state.videoId);
  $("#videoTitle").textContent = displayVideoTitle(document?.title || record?.title || "视频详情");
  const video = $("#detailVideo");
  const hasRequested = state.videoResumeTime !== null && Number.isFinite(Number(state.videoResumeTime));
  const requested = Number(state.videoResumeTime);
  if (hasRequested) {
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(Math.max(0, requested), Number.isFinite(video.duration) ? video.duration : requested);
      state.videoResumeTime = null;
      syncPlayerControls();
    }, {once: true});
  }
  video.src = `/api/videos/${state.videoId}/media`;
  const rows = chapterRows();
  $("#chapterCount").textContent = `${rows.length} 章`;
  const list = $("#chapterList");
  list.replaceChildren();
  const progress = $("#chapterProgress");
  progress.replaceChildren();
  const total = Number(state.detail?.artifacts?.raw?.duration_ms || rows.at(-1)?.end_ms || 1);

  rows.forEach((chapter, index) => {
    const image = node("img");
    image.loading = "lazy";
    image.alt = `${chapter.title} 截图`;
    image.src = `/api/videos/${state.videoId}/thumbnail?ms=${Math.max(0, Number(chapter.start_ms || 0) + 300)}`;
    const card = node("button", {className: `chapter-card${index === state.chapterIndex ? " active" : ""}`}, [
      node("div", {className: "chapter-thumb"}, [image, node("span", {className: "thumb-time", text: formatTime(chapter.start_ms)})]),
      node("div", {className: "chapter-copy"}, [
        node("span", {className: "chapter-number", text: `第 ${index + 1} 章`}),
        node("h3", {text: chapter.title || "未命名章节"}),
        node("div", {className: "chapter-duration", text: `时长 ${durationText(chapter.start_ms, chapter.end_ms)}`}),
        node("p", {className: "chapter-summary", text: chapter.summary || "暂无摘要"}),
      ]),
    ]);
    card.addEventListener("click", () => setChapter(index, true));
    list.append(card);

    const dot = node("button", {className: `chapter-dot${index === state.chapterIndex ? " active" : ""}`, title: chapter.title});
    dot.addEventListener("click", () => setChapter(index, true));
    progress.append(dot);
  });
  positionChapterDots(total);
  syncPlayerControls();

  const confirmed = document?.review_status === "confirmed";
  $("#reviewBadge").textContent = confirmed ? "已入库" : "待审核";
  $("#reviewBadge").className = `badge ${confirmed ? "ok" : "pending"}`;
  $("#confirmButton").disabled = confirmed || !document;
  $("#confirmButton").textContent = confirmed ? "已确认入库" : "确认入库";
  if (!rows.length) {
    $("#currentChapterTitle").textContent = "尚未生成语义片段";
    $("#documentTitle").textContent = "暂无章节";
    $("#chapterDocument").replaceChildren(node("p", {className: "muted", text: "请先回到列表勾选该视频并执行一键处理。"}));
    $("#editChapterButton").disabled = true;
  } else {
    $("#editChapterButton").disabled = false;
    renderChapterSelection();
  }
}

async function loadVideo(videoId, requestedChapter = 0) {
  state.videoId = videoId;
  state.detail = await api(`/api/videos/${videoId}`);
  state.chapterIndex = Math.max(0, Math.min(requestedChapter, chapterRows().length - 1));
  renderVideo();
}

function renderMarkdown(markdown, root) {
  root.replaceChildren();
  const lines = markdown.replace(/\r/g, "").split("\n");
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) root.append(node("p", {text: paragraphText(paragraph.join("\n"))}));
    paragraph = [];
  };
  for (const line of lines) {
    if (line.startsWith("### ")) { flush(); root.append(node("h3", {text: line.slice(4)})); }
    else if (line.startsWith("## ")) { flush(); root.append(node("h2", {text: line.slice(3)})); }
    else if (line.startsWith("> ")) { flush(); root.append(node("blockquote", {}, [node("div", {text: line.slice(2)})])); }
    else if (!line.trim()) flush();
    else paragraph.push(line);
  }
  flush();
}

function chapterSuggestions() {
  const chapter = currentChapter();
  if (!chapter) return [];
  const changes = state.detail?.artifacts?.suggestions || [];
  return changes.filter(item =>
    item?.status === "pending_review"
    && Number(item.sentence_id) >= Number(chapter.start_sentence_id)
    && Number(item.sentence_id) <= Number(chapter.end_sentence_id)
  );
}

function suggestionStatus(item) {
  if (item.decision === "confirmed_by_video") return ["随视频已确认", "ok"];
  if (item.decision === "approved" && item.approved_text && paragraphText(item.approved_text) !== paragraphText(item.proposed_text)) return ["已自定义", "ok"];
  if (item.decision === "approved") return ["已同意", "ok"];
  if (item.decision === "rejected") return ["已拒绝", "missing"];
  return ["AI 暂用，待复核", "pending"];
}

function renderSuggestionPanel() {
  const items = chapterSuggestions();
  const pending = items.filter(item => !["approved", "rejected", "confirmed_by_video"].includes(item.decision)).length;
  $("#suggestionCount").textContent = items.length ? `${items.length} 条 · ${pending} 条待处理` : "当前章节无待复核项";
  const root = $("#suggestionList");
  root.replaceChildren();
  if (!items.length) {
    root.append(node("div", {className: "suggestion-empty", text: "当前章节没有模型标记为不确定的清洗建议。"}));
    return;
  }
  for (const item of items) {
    const [statusText, statusClass] = suggestionStatus(item);
    const card = node("article", {className: "suggestion-card"});
    const listen = node("button", {className: "suggestion-listen", text: "▶ 回听"});
    listen.type = "button";
    listen.addEventListener("click", event => {
      event.stopPropagation();
      openEditorVideoAtSentence(item.sentence_id);
    });
    card.append(node("div", {className: "suggestion-card-head"}, [
      node("strong", {text: `句子 ${item.sentence_id}`}),
      listen,
      node("span", {className: `badge ${statusClass}`, text: statusText}),
    ]));
    card.append(
      node("div", {className: "suggestion-text"}, [node("span", {text: "原文"}), node("span", {text: item.raw_text || "（空）"})]),
      node("div", {className: "suggestion-text"}, [node("span", {text: "AI建议"}), node("span", {text: item.proposed_text || "（无建议）"})]),
      node("p", {className: "suggestion-reason", text: `不确定原因：${item.review_reason || "模型未说明"}`}),
    );
    if (!["approved", "rejected", "confirmed_by_video"].includes(item.decision)) {
      const approve = node("button", {className: "approve", text: "同意建议"});
      const reject = node("button", {className: "reject", text: "拒绝，保留原文"});
      const custom = node("button", {className: "custom", text: "自定义"});
      const customEditor = node("div", {className: "suggestion-custom-editor hidden"});
      const customText = node("textarea");
      customText.value = item.effective_text || item.proposed_text || item.raw_text || "";
      customText.placeholder = "结合原文和 AI 建议，填写最终采用的文字";
      const cancelCustom = node("button", {className: "custom-cancel", text: "取消"});
      const saveCustom = node("button", {className: "custom-save", text: "保存自定义"});
      cancelCustom.type = "button";
      saveCustom.type = "button";
      approve.addEventListener("click", event => {
        event.stopPropagation();
        decideSuggestion(item, "approved", null, approve, reject, custom);
      });
      reject.addEventListener("click", event => {
        event.stopPropagation();
        decideSuggestion(item, "rejected", null, approve, reject, custom);
      });
      custom.addEventListener("click", event => {
        event.stopPropagation();
        customEditor.classList.toggle("hidden");
        if (!customEditor.classList.contains("hidden")) customText.focus();
      });
      cancelCustom.addEventListener("click", event => {
        event.stopPropagation();
        customEditor.classList.add("hidden");
      });
      saveCustom.addEventListener("click", event => {
        event.stopPropagation();
        const value = paragraphText(customText.value);
        if (!value) return toast("自定义内容不能为空", true);
        decideSuggestion(item, "approved", value, approve, reject, custom, saveCustom);
      });
      customEditor.append(customText, node("div", {className: "suggestion-custom-actions"}, [cancelCustom, saveCustom]));
      card.append(node("div", {className: "suggestion-actions"}, [approve, reject, custom]), customEditor);
    }
    root.append(card);
  }
}

function mergeSuggestionDecisionIntoEditor(item, nextText) {
  const input = $("#chapterContentInput");
  const candidates = [item.effective_text, item.proposed_text, item.raw_text]
    .map(paragraphText).filter((value, index, values) => value && value !== nextText && values.indexOf(value) === index);
  const previous = candidates.find(value => input.value.includes(value));
  if (!previous) return false;
  input.value = input.value.replace(previous, nextText);
  renderEditorPreview();
  return true;
}

async function decideSuggestion(item, decision, approvedText = null, ...buttons) {
  for (const button of buttons) button.disabled = true;
  const wasDirty = state.markdownDirty;
  const nextText = paragraphText(decision === "rejected" ? item.raw_text : (approvedText || item.proposed_text || item.raw_text));
  try {
    state.detail = await api(`/api/videos/${state.videoId}/suggestions/${item.sentence_id}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        decision,
        approved_text: decision === "approved" ? nextText : null,
      }),
    });
    renderSuggestionPanel();
    if (!wasDirty) {
      fillEditorFields(currentChapter());
    } else {
      mergeSuggestionDecisionIntoEditor(item, nextText);
      state.markdownDirty = true;
    }
    toast(approvedText ? "自定义内容已保存并同步数据" : decision === "approved" ? "已同意 AI 建议并同步数据" : "已拒绝建议并恢复原文");
  } catch (error) {
    toast(error.message, true);
    for (const button of buttons) button.disabled = false;
  }
}

function singleParagraph(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function editorKeywords() {
  const values = $("#chapterKeywordsInput").value.split(/[,，、;；]+/);
  return [...new Set(values.map(singleParagraph).filter(value => value && value !== "无"))];
}

function editorMarkdown() {
  const chapter = currentChapter();
  if (!chapter) return "";
  const title = singleParagraph($("#chapterTitleInput").value) || "未命名章节";
  const keywords = editorKeywords();
  const summary = singleParagraph($("#chapterSummaryInput").value);
  const content = singleParagraph($("#chapterContentInput").value);
  return [
    `## ${title}`,
    "",
    `> 时间：${formatTime(chapter.start_ms)}–${formatTime(chapter.end_ms)}`,
    `> 关键词：${keywords.join("、") || "无"}`,
    "",
    "### 章节摘要",
    "",
    summary,
    "",
    "### 章节正文",
    "",
    content,
    "",
  ].join("\n");
}

function renderEditorPreview() {
  state.markdown = editorMarkdown();
  $("#editorTitle").textContent = singleParagraph($("#chapterTitleInput").value) || "未命名章节";
  renderMarkdown(state.markdown, $("#markdownPreview"));
}

function fillEditorFields(chapter) {
  if (!chapter) return;
  $("#chapterTimeDisplay").value = `${formatTime(chapter.start_ms)}–${formatTime(chapter.end_ms)}`;
  $("#chapterTitleInput").value = chapter.title || "";
  $("#chapterKeywordsInput").value = (chapter.keywords || []).join("、");
  $("#chapterSummaryInput").value = singleParagraph(chapter.summary);
  $("#chapterContentInput").value = singleParagraph(chapter.content);
  renderEditorPreview();
}

async function generateMetadata() {
  const button = $("#generateMetadataButton");
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "AI 生成中…";
  try {
    const result = await api(`/api/videos/${state.videoId}/segments/${state.chapterIndex + 1}/generate-metadata`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({markdown: editorMarkdown()}),
    });
    $("#chapterTitleInput").value = result.label.title || "";
    $("#chapterKeywordsInput").value = (result.label.keywords || []).join("、");
    $("#chapterSummaryInput").value = result.label.summary || "";
    state.markdownDirty = true;
    renderEditorPreview();
    toast("新标题、关键词和摘要已填入，请确认后保存");
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function loadEditor(videoId, chapterIndex) {
  state.videoId = videoId;
  state.detail = await api(`/api/videos/${videoId}`);
  state.chapterIndex = Math.max(0, Math.min(chapterIndex, chapterRows().length - 1));
  const chapter = currentChapter();
  if (!chapter) throw new Error("该视频还没有可编辑章节");
  state.markdownDirty = false;
  fillEditorFields(chapter);
  renderSuggestionPanel();
  configureEditorVideo();
}

function editorDetailPath() {
  return `/videos/${state.videoId}?chapter=${state.chapterIndex + 1}`;
}

function requestEditorReturn() {
  if (!state.markdownDirty) return navigate(editorDetailPath());
  pauseEditorVideo();
  $("#unsavedEditModal").classList.remove("hidden");
}

function closeUnsavedEditModal() {
  $("#unsavedEditModal").classList.add("hidden");
}

function closeGlossaryModal() {
  $("#glossaryModal").classList.add("hidden");
  state.glossaryCandidates = [];
  $("#saveMarkdownButton").disabled = false;
}

function renderGlossaryModal(candidates) {
  state.glossaryCandidates = candidates;
  const root = $("#glossaryCandidateList");
  root.replaceChildren();
  candidates.forEach((candidate, index) => {
    const checkbox = node("input");
    checkbox.type = "checkbox";
    checkbox.className = "glossary-select";
    checkbox.dataset.index = String(index);
    const scope = node("select", {className: "glossary-scope"});
    scope.dataset.index = String(index);
    scope.append(
      node("option", {text: "仅适用本视频"}),
      node("option", {text: "全局词表"}),
    );
    scope.options[0].value = "video";
    scope.options[1].value = "global";
    scope.value = "video";
    const termInput = node("input", {className: "glossary-term-input"});
    termInput.type = "text";
    termInput.maxLength = 40;
    termInput.value = candidate.term;
    termInput.dataset.index = String(index);
    termInput.setAttribute("aria-label", "要加入词表的词");
    const aliasInput = node("input", {className: "glossary-term-input glossary-alias-input"});
    aliasInput.type = "text";
    aliasInput.maxLength = 200;
    aliasInput.value = (candidate.aliases || []).join("、");
    aliasInput.dataset.index = String(index);
    aliasInput.setAttribute("aria-label", "错误举例");
    const selectRow = () => { checkbox.checked = true; };
    termInput.addEventListener("input", selectRow);
    aliasInput.addEventListener("input", selectRow);
    scope.addEventListener("change", selectRow);
    root.append(node("div", {className: "glossary-candidate"}, [
      checkbox,
      node("div", {className: "glossary-term"}, [
        node("div", {className: "glossary-edit-grid"}, [
          node("label", {className: "glossary-edit-field"}, [node("span", {text: "标准词"}), termInput]),
          node("label", {className: "glossary-edit-field"}, [node("span", {text: "错误举例"}), aliasInput]),
        ]),
      ]),
      scope,
    ]));
  });
  $("#glossaryModal").classList.remove("hidden");
}

function selectedGlossaryTerms() {
  return [...document.querySelectorAll(".glossary-select:checked")].map(checkbox => {
    const index = Number(checkbox.dataset.index);
    const candidate = state.glossaryCandidates[index];
    const scope = document.querySelector(`.glossary-scope[data-index="${index}"]`).value;
    const term = document.querySelector(`.glossary-term-input[data-index="${index}"]`).value.trim();
    const aliasText = document.querySelector(`.glossary-alias-input[data-index="${index}"]`).value;
    const aliases = [...new Set(aliasText.split(/[,，、;；]+/).map(value => value.trim()).filter(value => value && value !== term))];
    if (!term) throw new Error("勾选的入库词不能为空");
    return {term, aliases, scope};
  });
}

function finishGlossaryApplyFlow() {
  $("#glossaryApplyModal").classList.add("hidden");
  state.pendingGlossaryTerms = [];
  state.glossaryHits = [];
  state.glossaryApplyScope = null;
  navigate(`/videos/${state.videoId}?chapter=${state.chapterIndex + 1}`);
}

function showGlossaryApplyChoice(terms) {
  state.pendingGlossaryTerms = terms;
  state.glossaryHits = [];
  state.glossaryApplyScope = null;
  $("#glossaryApplyChoices").classList.remove("hidden");
  $("#glossarySearchResults").classList.add("hidden");
  $("#glossaryApplyModal").classList.remove("hidden");
}

function renderGlossaryHits(hits, scope) {
  state.glossaryHits = hits;
  state.glossaryApplyScope = scope;
  $("#glossaryApplyChoices").classList.add("hidden");
  $("#glossarySearchResults").classList.remove("hidden");
  $("#glossarySearchSummary").textContent = hits.length
    ? `找到 ${hits.length} 句话。请取消不需要修改的项目，再应用所选。`
    : "没有找到词表中记录的错误写法。";
  $("#glossaryHitList").replaceChildren(...hits.map(hit => {
    const checkbox = node("input");
    checkbox.type = "checkbox";
    checkbox.className = "glossary-hit-select";
    checkbox.value = hit.hit_id;
    checkbox.checked = true;
    return node("label", {className: "glossary-hit"}, [
      checkbox,
      node("div", {}, [
        node("div", {className: "glossary-hit-meta"}, [
          node("span", {text: hit.video_title}),
          node("span", {text: formatTime(hit.start_ms)}),
          node("span", {className: "glossary-hit-change", text: `${hit.alias} → ${hit.term}`}),
        ]),
        node("div", {className: "glossary-hit-text", text: hit.snippet}),
      ]),
    ]);
  }));
  $("#applySelectedGlossaryHits").disabled = !hits.length;
}

async function searchGlossaryHits(scope) {
  for (const button of [$("#applyGlossaryDatabase"), $("#applyGlossaryVideo"), $("#skipGlossaryApply")]) button.disabled = true;
  try {
    const result = await api("/api/glossary/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({source_video_id: state.videoId, terms: state.pendingGlossaryTerms, scope}),
    });
    renderGlossaryHits(result.hits || [], scope);
  } catch (error) {
    toast(error.message, true);
  } finally {
    for (const button of [$("#applyGlossaryDatabase"), $("#applyGlossaryVideo"), $("#skipGlossaryApply")]) button.disabled = false;
  }
}

async function applySelectedGlossaryHits() {
  const selectedHitIds = [...document.querySelectorAll(".glossary-hit-select:checked")].map(input => input.value);
  if (!selectedHitIds.length) return finishGlossaryApplyFlow();
  const button = $("#applySelectedGlossaryHits");
  button.disabled = true;
  try {
    const result = await api("/api/glossary/apply", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source_video_id: state.videoId,
        terms: state.pendingGlossaryTerms,
        scope: state.glossaryApplyScope,
        selected_hit_ids: selectedHitIds,
      }),
    });
    toast(`已更新 ${result.changed_videos} 个视频、${result.changed_sentences} 句话`);
    finishGlossaryApplyFlow();
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
  }
}

async function persistMarkdown(glossaryTerms) {
  const button = $("#saveMarkdownButton");
  button.disabled = true;
  $("#saveWithoutGlossary").disabled = true;
  $("#saveWithGlossary").disabled = true;
  try {
    state.detail = await api(`/api/videos/${state.videoId}/segments/${state.chapterIndex + 1}`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        title: singleParagraph($("#chapterTitleInput").value),
        summary: singleParagraph($("#chapterSummaryInput").value),
        content: singleParagraph($("#chapterContentInput").value),
        keywords: editorKeywords(),
        glossary_terms: glossaryTerms,
        expected_revision: Number(state.detail?.artifacts?.segments?.revision || 0),
      }),
    });
    state.markdownDirty = false;
    $("#glossaryModal").classList.add("hidden");
    const termMessage = glossaryTerms.length ? `，${glossaryTerms.length} 个词已加入词表` : "";
    toast(`编辑已保存，JSON 与 Markdown 已同步更新${termMessage}`);
    if (glossaryTerms.length) showGlossaryApplyChoice(glossaryTerms);
    else navigate(`/videos/${state.videoId}?chapter=${state.chapterIndex + 1}`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    $("#saveWithoutGlossary").disabled = false;
    $("#saveWithGlossary").disabled = false;
  }
}

async function saveMarkdown() {
  const button = $("#saveMarkdownButton");
  button.disabled = true;
  try {
    const result = await api(`/api/videos/${state.videoId}/segments/${state.chapterIndex + 1}/glossary-candidates`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({markdown: editorMarkdown()}),
    });
    if (result.candidates?.length) {
      renderGlossaryModal(result.candidates);
      return;
    }
    await persistMarkdown([]);
  } catch (error) {
    toast(error.message, true);
  } finally {
    if ($("#glossaryModal").classList.contains("hidden")) button.disabled = false;
  }
}

async function confirmVideo() {
  if (!confirm("确认当前章节内容无误并入库吗？")) return;
  const button = $("#confirmButton");
  button.disabled = true;
  try {
    state.detail = await api(`/api/videos/${state.videoId}/confirm`, {method: "POST"});
    toast("已确认入库");
    await loadDashboard();
    renderVideo();
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
  }
}

async function route() {
  const editMatch = location.pathname.match(/^\/videos\/([0-9a-f]{64})\/edit\/?$/);
  const videoMatch = location.pathname.match(/^\/videos\/([0-9a-f]{64})\/?$/);
  const chapter = Math.max(0, Number(new URLSearchParams(location.search).get("chapter") || 1) - 1);
  if (editMatch) {
    showView("editorView");
    await loadEditor(editMatch[1], chapter);
  } else if (videoMatch) {
    showView("videoView");
    await loadVideo(videoMatch[1], chapter);
  } else {
    showView("dashboardView");
    await loadDashboard();
    await loadJobs();
  }
}

$("#uploadButton").addEventListener("click", uploadFiles);
$("#apiConnectionButton").addEventListener("click", openApiDialog);
$("#closeApiDialog").addEventListener("click", () => $("#apiDialog").close());
$("#apiForm").addEventListener("submit", submitApiConnection);
$("#disconnectApiButton").addEventListener("click", disconnectApiConnection);
$("#processButton").addEventListener("click", startProcessing);
$("#cancelTaskButton").addEventListener("click", cancelActiveJob);
$("#searchInput").addEventListener("input", renderRows);
$("#selectAll").addEventListener("change", event => {
  for (const row of visibleRecords()) event.target.checked ? state.selected.add(row.id) : state.selected.delete(row.id);
  renderRows();
});
$("#refreshButton").addEventListener("click", () => route().catch(error => toast(error.message, true)));
$("#confirmButton").addEventListener("click", confirmVideo);
$("#editChapterButton").addEventListener("click", () => {
  const video = $("#detailVideo");
  state.videoResumeTime = Number.isFinite(video.currentTime) ? video.currentTime : null;
  navigate(`/videos/${state.videoId}/edit?chapter=${state.chapterIndex + 1}`);
});
$("#editorBackButton").addEventListener("click", () => {
  const video = $("#editorVideo");
  state.videoResumeTime = Number.isFinite(video.currentTime) ? video.currentTime : null;
  requestEditorReturn();
});
$("#closeUnsavedEditModal").addEventListener("click", closeUnsavedEditModal);
$("#discardEditButton").addEventListener("click", () => {
  state.markdownDirty = false;
  closeUnsavedEditModal();
  navigate(editorDetailPath());
});
$("#saveAndReturnButton").addEventListener("click", () => {
  closeUnsavedEditModal();
  saveMarkdown();
});
$("#playChapterButton").addEventListener("click", openEditorVideoAtChapterStart);
$("#saveMarkdownButton").addEventListener("click", saveMarkdown);
$("#closeGlossaryModal").addEventListener("click", closeGlossaryModal);
$("#saveWithoutGlossary").addEventListener("click", () => persistMarkdown([]));
$("#saveWithGlossary").addEventListener("click", () => {
  try { persistMarkdown(selectedGlossaryTerms()); }
  catch (error) { toast(error.message, true); }
});
$("#applyGlossaryDatabase").addEventListener("click", () => searchGlossaryHits("database"));
$("#applyGlossaryVideo").addEventListener("click", () => searchGlossaryHits("video"));
$("#skipGlossaryApply").addEventListener("click", finishGlossaryApplyFlow);
$("#cancelGlossaryHits").addEventListener("click", finishGlossaryApplyFlow);
$("#applySelectedGlossaryHits").addEventListener("click", applySelectedGlossaryHits);
$("#generateMetadataButton").addEventListener("click", generateMetadata);
$("#editorPlayButton").addEventListener("click", toggleEditorPlayback);
$("#editorVideo").addEventListener("click", toggleEditorPlayback);
$("#editorBack5Button").addEventListener("click", () => {
  const video = $("#editorVideo");
  seekEditorVideo(Math.max(0, video.currentTime - 5), true);
});
$("#editorForward5Button").addEventListener("click", () => {
  const video = $("#editorVideo");
  seekEditorVideo(Math.min(video.duration, video.currentTime + 5), true);
});
$("#editorVideoCloseButton").addEventListener("click", closeEditorVideo);
$("#editorFullscreenButton").addEventListener("click", () => {
  const panel = document.querySelector(".editor-video-panel");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else panel.requestFullscreen().catch(() => {});
});
$("#editorVideoSeek").addEventListener("input", event => {
  const video = $("#editorVideo");
  if (Number.isFinite(video.duration) && video.duration > 0) {
    seekEditorVideo(Number(event.target.value) / 1000 * video.duration, false);
  }
});
$("#editorVideo").addEventListener("loadedmetadata", syncEditorPlayerControls);
$("#editorVideo").addEventListener("play", syncEditorPlayerControls);
$("#editorVideo").addEventListener("pause", syncEditorPlayerControls);
$("#editorVideo").addEventListener("ended", syncEditorPlayerControls);
$("#editorVideo").addEventListener("timeupdate", event => {
  syncEditorPlayerControls();
});
for (const selector of ["#chapterTitleInput", "#chapterKeywordsInput", "#chapterSummaryInput", "#chapterContentInput"]) {
  $(selector).addEventListener("input", () => {
    state.markdownDirty = true;
    renderEditorPreview();
  });
}
$("#playButton").addEventListener("click", togglePlayback);
$("#detailVideo").addEventListener("click", togglePlayback);
$("#videoSeek").addEventListener("input", event => {
  const video = $("#detailVideo");
  if (Number.isFinite(video.duration) && video.duration > 0) {
    video.currentTime = Number(event.target.value) / 1000 * video.duration;
  }
  syncPlayerControls();
});
$("#volumeControl").addEventListener("input", event => {
  $("#detailVideo").volume = Number(event.target.value);
});
$("#fullscreenButton").addEventListener("click", () => {
  const shell = document.querySelector(".video-shell");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  else shell.requestFullscreen().catch(() => {});
});
$("#detailVideo").addEventListener("loadedmetadata", event => {
  positionChapterDots(event.target.duration * 1000);
  syncPlayerControls();
});
$("#detailVideo").addEventListener("play", syncPlayerControls);
$("#detailVideo").addEventListener("pause", syncPlayerControls);
$("#detailVideo").addEventListener("ended", syncPlayerControls);
$("#detailVideo").addEventListener("timeupdate", event => {
  syncPlayerControls();
  const ms = event.target.currentTime * 1000;
  const index = chapterRows().findIndex(chapter => ms >= Number(chapter.start_ms) && ms < Number(chapter.end_ms));
  if (index >= 0 && index !== state.chapterIndex) setChapter(index, false);
});
document.querySelectorAll("[data-back]").forEach(button => button.addEventListener("click", () => navigate("/")));
document.querySelectorAll("[data-nav]").forEach(link => link.addEventListener("click", event => { event.preventDefault(); navigate(link.pathname); }));
$("#shutdownButton").addEventListener("click", async () => {
  if (!confirm("确定关闭视频处理服务吗？")) return;
  try {
    await api("/api/system/shutdown", {method: "POST"});
    document.body.innerHTML = "<main class='page'><section class='panel'><h2>服务正在关闭</h2><p>可以关闭此浏览器页面。</p></section></main>";
  } catch (error) { toast(error.message, true); }
});
window.addEventListener("popstate", () => route().catch(error => toast(error.message, true)));
window.addEventListener("beforeunload", event => {
  if (!state.markdownDirty || !location.pathname.endsWith("/edit")) return;
  event.preventDefault();
  event.returnValue = "";
});
window.addEventListener("keydown", event => {
  if (event.key === "Escape" && !$("#glossaryModal").classList.contains("hidden")) closeGlossaryModal();
});

Promise.all([route(), refreshCloudStatus()]).catch(error => toast(error.message, true));
setInterval(() => {
  if (location.pathname === "/") loadJobs();
}, 2500);
