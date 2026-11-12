import { useEffect, useRef, useState } from "react";
import { backend, request } from "../api";
import { countUnreviewedExclusions, needsExclusionReview } from "../maskPolicy.mjs";
import "./SequencePanel.css";

function CameraPath({ points }) {
  if (points.length < 2) return null;
  const bounds = [0, 1, 2].map((axis) => {
    const values = points.map((point) => point[axis]);
    return { axis, min: Math.min(...values), span: Math.max(...values) - Math.min(...values) };
  }).sort((a, b) => b.span - a.span);
  const [x, y] = bounds;
  const scale = Math.min(152 / (x.span || 1), 52 / (y.span || 1));
  const path = points.map((point) =>
    (10 + (point[x.axis] - x.min) * scale) + "," + (63 - (point[y.axis] - y.min) * scale)
  ).join(" ");
  return <svg viewBox="0 0 172 73" role="img" aria-label="Solved camera path projection">
    <polyline points={path} fill="none" stroke="#5ac1f8" strokeWidth="2" />
  </svg>;
}

export default function SequencePanel({
  onSequence, sequence, currentFrame, prompts, tool, setTool, onPrompt, setManifest, manifest,
  onSolve, busy: appBusy, setBusy, showTracks, setShowTracks, showMask, setShowMask, onSelectSourceFrame,
  projects, onOpenProject, projectError, onDiscardPrompts,
}) {
  const fileInput = useRef(null);
  const [fps, setFps] = useState("");
  const [stride, setStride] = useState(1);
  const [maxFrames, setMaxFrames] = useState(24);
  const [initialFocal, setInitialFocal] = useState("");
  const [exportFps, setExportFps] = useState("");
  const [samFrames, setSamFrames] = useState(24);
  const [sam, setSam] = useState(false);
  const [error, setError] = useState("");
  const [job, setJob] = useState(null);
  const [solve, setSolve] = useState(null);
  const [exportUrl, setExportUrl] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [preflight, setPreflight] = useState(null);
  const [savedSolves, setSavedSolves] = useState([]);
  const [savedExports, setSavedExports] = useState([]);
  const [revisions, setRevisions] = useState([]);
  const [recovering, setRecovering] = useState(Boolean(sequence));
  const busy = appBusy || recovering;
  const root = sequence ? "/sequences/" + sequence.sequence_id : null;
  const currentMask = manifest?.frames.find((frame) => frame.frame_number === currentFrame?.frame_number);
  const revisionKey = "matchmove.mask." + sequence?.sequence_id;
  const solveKey = "matchmove.solve." + sequence?.sequence_id;
  const canSetSolveLimit = Number.isInteger(Number(maxFrames)) && Number(maxFrames) >= 3 && Number(maxFrames) <= 120;
  const hasValidFocal = initialFocal === "" || (Number.isFinite(Number(initialFocal)) && Number(initialFocal) > 0);
  const canSolve = !busy && canSetSolveLimit && hasValidFocal && (sequence?.frames.length ?? 0) >= 3;

  function selectManifest(next) {
    setManifest(next);
    localStorage.setItem(revisionKey, next.mask_set_id);
  }

  function selectSolve(next) {
    setSolve({ ...next.result, jobId: next.job_id });
    localStorage.setItem(solveKey, next.job_id);
    onSolve(next.result);
  }

  // A page refresh must not force the user to rerun expensive camera solves/SAM.
  useEffect(() => {
    if (!root) return;
    let cancelled = false;
    async function recover() {
      try {
        const history = await request(root + "/jobs");
        if (cancelled) return;
        const entries = history.jobs ?? [];
        const solves = entries.filter((item) => item.operation === "solve" && item.status === "done" && item.result?.tracks_url);
        const exports = entries.filter((item) => item.operation === "export" && item.status === "done" && item.result?.export_url);
        setSavedSolves(solves);
        setSavedExports(exports);
        const latest = (operation) => entries.find((item) => item.operation === operation && item.status === "done");
        const solveJob = solves.find((item) => item.job_id === localStorage.getItem(solveKey)) ?? solves[0];
        if (solveJob) selectSolve(solveJob);
        const comparisonJob = latest("compare");
        if (comparisonJob?.result) setComparison({ ...comparisonJob.result, jobId: comparisonJob.job_id });
        const exportJob = exports.find((item) => item.result.source_job_id === solveJob?.job_id ||
          (solves.length === 1 && !item.result.source_job_id));
        if (exportJob?.result?.export_url) setExportUrl(exportJob.result.export_url);
        setPreflight(latest("preflight")?.result ?? null);
        const active = entries.find((item) => ["running", "queued"].includes(item.status));
        if (active) {
          setJob(active);
          setBusy(true);
        } else if (["failed", "cancelled"].includes(entries[0]?.status)) {
          setJob(entries[0]);
        }
        // Saved results remain usable if a mask manifest cannot be restored.
        const savedMasks = await request(root + "/foreground");
        if (cancelled) return;
        const items = savedMasks.mask_sets ?? [];
        setRevisions(items);
        const selected = items.find((item) => item.mask_set_id === localStorage.getItem(revisionKey)) ?? items[0];
        if (selected) {
          const saved = await request(selected.manifest_url);
          if (!cancelled) selectManifest(saved);
        }
      } catch (failure) {
        if (!cancelled) setError("Could not restore all saved results. " + failure.message);
      } finally {
        if (!cancelled) setRecovering(false);
      }
    }
    recover();
    return () => { cancelled = true; };
  }, [root]);

  async function reviewRegion(index, decision) {
    if (!manifest?.manifest_url || !currentFrame) return;
    setBusy(true);
    setError("");
    try {
      const updated = await request(manifest.manifest_url + "/review", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decisions: [{ frame_number: currentFrame.frame_number, region_index: index, decision }] }),
      });
      selectManifest(updated);
      setRevisions((items) => [updated, ...items]);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!job?.status_url || !["queued", "running"].includes(job.status)) return;
    let cancelled = false;
    let timer;
    async function poll() {
      try {
        const update = await request(job.status_url);
        if (cancelled) return;
        setError("");
        setJob({ ...update, status_url: job.status_url });
        if (["queued", "running"].includes(update.status)) {
          timer = setTimeout(poll, 900);
          return;
        }
        setBusy(false);
        if (update.status !== "done") return;
        const result = update.result;
        if (update.operation === "foreground") {
          selectManifest(result);
          setRevisions((items) => [result, ...items]);
          onDiscardPrompts();
          setShowMask(true);
        } else if (update.operation === "solve") {
          selectSolve(update);
          setExportUrl(null);
          setSavedSolves((items) => [update, ...items.filter((item) => item.job_id !== update.job_id)]);
        } else if (update.operation === "export") {
          setExportUrl(result.export_url);
          setSavedExports((items) => [update, ...items.filter((item) => item.job_id !== update.job_id)]);
        } else if (update.operation === "compare") {
          setComparison({ ...result, jobId: update.job_id });
        } else if (update.operation === "preflight") {
          setPreflight(result);
        }
      } catch (failure) {
        if (!cancelled) {
          setError(failure.message);
          timer = setTimeout(poll, 2500);
        }
      }
    }
    timer = setTimeout(poll, 400);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [job?.status_url]);

  async function upload(files) {
    if (!files?.length) return;
    setBusy(true);
    setError("");
    setJob(null);
    try {
      const form = new FormData();
      for (const file of files) form.append("files", file);
      if (fps.trim()) form.append("fps", fps.trim());
      form.append("video_stride", stride);
      const result = await request("/upload/", { method: "POST", body: form });
      onSequence(result);
      setSolve(null);
      setExportUrl(null);
      setComparison(null);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function start(operation, options = {}) {
    setBusy(true);
    setError("");
    setJob(null);
    try {
      const result = await request(root + "/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          operation,
          ...(["preflight", "solve", "compare"].includes(operation) && !options.comparison_job_id ? {
            max_frames: Number(maxFrames),
            initial_focal_px: initialFocal ? Number(initialFocal) : null,
          } : {}),
          ...(operation === "foreground" && options.foreground?.discover ? { sam_analysis_frames: Number(samFrames) } : {}),
          ...options,
        }),
      });
      setJob(result);
    } catch (failure) {
      setBusy(false);
      setError(failure.message);
    }
  }

  async function cancel() {
    try {
      const status = await request(job.status_url + "/cancel", { method: "POST" });
      setJob({ ...status, status_url: job.status_url });
    } catch (failure) { setError(failure.message); }
  }

  async function retry() {
    setBusy(true);
    setError("");
    try {
      setJob(await request(job.status_url + "/retry", { method: "POST" }));
    } catch (failure) {
      setError(failure.message);
      setBusy(false);
    }
  }

  function startSolve() {
    if (sam && !manifest?.masked_frames) {
      setError("Select a reviewed revision with an exclusion, or choose Baseline.");
      return;
    }
    // A false-positive foreground mask removes valid background features from SfM.
    // Never silently apply automatic exclusions that the user has not reviewed.
    if (sam && unreviewedExclusions > 0) {
      setError(`${unreviewedExclusions} automatically excluded region(s) still need review. Visit those frames and choose Exclude or Keep. Or disable SAM for a baseline solve.`);
      return;
    }
    start("solve", {
      sam, mask_set_id: sam ? manifest.mask_set_id : null,
    });
  }

  const statistics = solve?.statistics;
  const frameQuality = solve?.frame_quality?.[currentFrame?.frame_number];
  const currentSampled = statistics?.selected_frame_numbers?.includes(currentFrame?.frame_number);
  const candidateFrames = (manifest?.frames ?? []).filter((frame) => frame.regions.length > 0);
  const unreviewedExclusions = countUnreviewedExclusions(manifest);
  const pendingFrames = (manifest?.frames ?? []).filter((frame) => frame.regions.some(
    needsExclusionReview
  ));
  function nextMaskFrame(frames) {
    const later = frames.find((frame) => frame.frame_number > (currentFrame?.frame_number ?? -Infinity));
    const next = later ?? frames[0];
    if (next) onSelectSourceFrame(next.frame_number);
  }
  return <aside className="sequence-panel">
    <section>
      <h2>Footage</h2>
      <label>Saved project <select value={sequence?.sequence_id ?? ""} disabled={busy}
        onChange={(e) => { if (e.target.value) onOpenProject(e.target.value); }}>
        <option value="">Select footage</option>
        {projects.map((item) => <option key={item.sequence_id} value={item.sequence_id}>{item.source_name} · {item.frame_count} frames</option>)}
      </select></label>
      {projectError && <div role="alert" className="error">{projectError}</div>}
      <label>Sequence FPS <input value={fps} placeholder="Optional" disabled={busy} onChange={(e) => setFps(e.target.value)} /></label>
      <label>Video sampling step <input type="number" min="1" max="1200" value={stride} disabled={busy} onChange={(e) => setStride(e.target.value)} /></label>
      <button disabled={busy} onClick={() => fileInput.current.click()}>Upload video or sequence</button>
      <input hidden ref={fileInput} type="file" multiple accept="video/*,image/*,.exr,.mkv,.mxf,.mts" onChange={(e) => upload(e.target.files)} />
      {sequence && <div className="metadata">
        {sequence.source_name && <strong>{sequence.source_name}</strong>}
        <span>{sequence.frames.length} available{sequence.source_frame_count ? " / " + sequence.source_frame_count + " source frames" : " frames"}</span>
        <span>{sequence.resolution ? sequence.resolution.width + " × " + sequence.resolution.height : "Mixed resolution"} · {sequence.fps ? sequence.fps.toFixed(3) + " fps" : "FPS unknown"}</span>
        {sequence.failed_uploads.map((file, index) => <span key={index} className="error">{file.source_name} · {file.error}</span>)}
      </div>}
    </section>
    <section>
      <h2>Foreground</h2>
      {revisions.length > 0 && <label>Saved revision <select value={manifest?.mask_set_id ?? ""} disabled={busy}
        onChange={async (e) => {
          const revision = revisions.find((item) => item.mask_set_id === e.target.value);
          if (!revision) return;
          setBusy(true);
          try { selectManifest(await request(revision.manifest_url)); }
          catch (failure) { setError(failure.message); }
          finally { setBusy(false); }
        }}>
        <option value="" disabled>Select revision</option>
        {revisions.map((item) => <option key={item.mask_set_id} value={item.mask_set_id}>{item.mask_set_id.slice(0, 8)} · {item.masked_frames} with exclusions</option>)}
      </select></label>}
      {prompts.length > 0 && <button disabled={busy} onClick={onDiscardPrompts}>Discard unsaved prompts</button>}
      <label className="check"><input type="checkbox" checked={sam} disabled={busy} onChange={(e) => setSam(e.target.checked)} /> Use SAM-assisted masking</label>
      <label>Prompt tool <select value={tool} onChange={(e) => setTool(e.target.value)} disabled={!currentFrame || busy}>
        <option value="none">Navigate</option><option value="box">Draw box</option>
        <option value="foreground">Add foreground point</option><option value="background">Add background point</option>
      </select></label>
      <label>Region decision <select value={prompts.find((prompt) => prompt.frame_number === currentFrame?.frame_number)?.decision ?? "auto"}
        onChange={(e) => onPrompt({ decision: e.target.value })} disabled={!currentFrame || busy}>
        <option value="auto">Check motion</option><option value="keep">Keep features</option><option value="exclude">Exclude features</option>
      </select></label>
      <button disabled={busy || !currentFrame} onClick={() => onPrompt(null)}>Clear current prompt</button>
      <button disabled={busy || !prompts.some((prompt) => prompt.frame_number === currentFrame?.frame_number &&
        (prompt.box || prompt.points.length))} onClick={() => start("foreground", { foreground: {
        prompts: prompts.filter((prompt) => prompt.box || prompt.points.length),
      } })}>Segment prompts only</button>
      <label>SAM analysis frames <input type="number" min="1" max="120" value={samFrames}
        disabled={busy} onChange={(e) => setSamFrames(e.target.value)} /></label>
      <span>Separate from camera-solve frames. More frames require additional motion analysis and possibly SAM inference.</span>
      <button disabled={busy || !sequence?.frames.length || !Number.isInteger(Number(samFrames)) || Number(samFrames) < 1 || Number(samFrames) > 120} onClick={() => start("foreground", { foreground: {
        discover: true, prompts: prompts.filter((prompt) => prompt.box || prompt.points.length),
      } })}>Generate and review masks</button>
      <label className="check"><input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} /> Show mask preview</label>
      {manifest && <span>{manifest.masked_frames} with exclusions / {manifest.analyzed_frames ?? manifest.frames.length} analyzed
        {manifest.frames_with_proposals != null && ` · ${manifest.frames_with_proposals} with proposals`}</span>}
      {manifest?.frame_status_counts && <span>
        {manifest.frame_status_counts.no_proposal} without proposals · {manifest.frame_status_counts.no_approved_exclusion} proposed but not approved
      </span>}
      {currentMask?.analyzed && currentMask.mask_status && <span>
        {currentMask.regions.some(needsExclusionReview) ? "This frame has a proposed exclusion" : currentMask.mask_status.replaceAll("_", " ")}
      </span>}
      {manifest?.frames_with_proposals > 0 && <span>
        Auto motion masks are suggestions, not confirmed moving objects.
        {unreviewedExclusions > 0 ? ` ${unreviewedExclusions} excluded region(s) need review before a masked solve.` : " All automatic exclusions have been reviewed."}
      </span>}
      {candidateFrames.length > 0 && <button disabled={busy} onClick={() => { setShowMask(true); nextMaskFrame(candidateFrames); }}>
        Next candidate frame ({candidateFrames.length} frames)
      </button>}
      {unreviewedExclusions > 0 && <button disabled={busy} onClick={() => { setShowMask(true); nextMaskFrame(pendingFrames); }}>
        Next unreviewed exclusion ({pendingFrames.length} frames)
      </button>}
      {currentMask?.regions.map((region, index) => <div key={index} className="mask-review-region">
        <span>Region {index + 1} · {needsExclusionReview(region) ? "review required" : (region.decision?.replaceAll("_", " ") ?? "candidate")}
          {region.review_decision && " (reviewed)"}</span>
        {region.motion?.map((item) => <span key={item.neighbor}>
          Frame {item.neighbor} · {item.tracked_points} tracks · {item.decision}
        </span>)}
        <button disabled={busy} onClick={() => reviewRegion(index, "exclude")}>Exclude region</button>
        <button disabled={busy} onClick={() => reviewRegion(index, "keep")}>Keep region</button>
        <button disabled={busy} onClick={() => onPrompt(region.prompt)}>Edit prompt (new SAM run)</button>
      </div>)}
      {currentMask?.mask_url && <div className="downloads">
        <a href={backend + currentMask.mask_url} download>Exclusion PNG</a>
        <a href={backend + currentMask.candidate_url} download>Candidate PNG</a>
      </div>}
    </section>
    <section>
      <h2>Camera solve</h2>
      <button disabled={!canSolve} onClick={() => start("preflight")}>Check solve geometry</button>
      {preflight && <div className="metadata">
        <span>{preflight.statistics.verified_pairs} verified pairs · {preflight.geometry.frames_with_verified_neighbors} connected frames</span>
        <a href={backend + preflight.report_url}>Download preflight report</a>
      </div>}
      <label>Maximum solve frames <input type="number" min="3" max="120" value={maxFrames} disabled={busy} onChange={(e) => setMaxFrames(e.target.value)} /></label>
      {!canSetSolveLimit && <span>Enter a solve limit from 3 to 120 frames.</span>}
      <label>Initial focal length (px) <input type="number" min="0.01" step="any" value={initialFocal} placeholder="Automatic"
        disabled={busy} onChange={(e) => setInitialFocal(e.target.value)} /></label>
      {!hasValidFocal && <span>Enter a positive focal length in source pixels.</span>}
      <button className="primary" disabled={!canSolve} onClick={startSolve}>Start Camera Solve</button>
      <button disabled={!canSolve || !manifest?.masked_frames || unreviewedExclusions > 0} onClick={() => start("compare", { mask_set_id: manifest.mask_set_id })}>Compare baseline and masked</button>
      {busy && <div role="status">
        <span>{recovering ? "Restoring project…" : job?.stage ?? "Uploading or starting job…"}</span>
        {job?.total ? <progress value={job.completed ?? 0} max={job.total} /> : <progress />}
        {["queued", "running"].includes(job?.status) && <button disabled={job.cancellation_requested} onClick={cancel}>{job.cancellation_requested ? "Cancelling…" : "Cancel"}</button>}
      </div>}
      {!busy && job?.status === "cancelled" && <span role="status">Job cancelled.</span>}
      {error && <div role="alert" className="error">{error}</div>}
      {job?.status === "failed" && <div role="alert" className="error">{job.error}</div>}
      {!busy && job?.parameters && ["failed", "cancelled"].includes(job.status) &&
        <button onClick={retry}>Retry {job.operation}</button>}
      {job?.status === "failed" && job.measurements && <div className="metadata">
        <span>Failed during {job.failed_stage?.replaceAll("_", " ")}</span>
        {job.measurements.verified_pairs != null && <span>{job.measurements.verified_pairs} verified pairs</span>}
        <a href={backend + job.status_url + "/files/diagnostics.json"}>Download solve diagnostics</a>
      </div>}
    </section>
    {statistics && <section>
      <h2>Results</h2>
      {savedSolves.length > 1 && <label>Saved solve <select value={solve.jobId} disabled={busy} onChange={(e) => {
        const selected = savedSolves.find((item) => item.job_id === e.target.value);
        if (!selected) return;
        selectSolve(selected);
        setExportUrl(savedExports.find((item) => item.result.source_job_id === selected.job_id)?.result.export_url ?? null);
      }}>
        {savedSolves.map((item) => <option key={item.job_id} value={item.job_id}>
          {item.job_id.slice(0, 8)} · {item.result.masking_applied ? "masked" : "baseline"}
        </option>)}
      </select></label>}
      <span>{statistics.registered_frames}/{statistics.input_frames} sampled frames solved · {solve.status}</span>
      <span>{statistics.unregistered_frames} sampled frames unregistered · {solve.source_frame_count} source frames</span>
      <span>{statistics.sparse_points.toLocaleString()} sparse points</span>
      <span>{statistics.mean_reprojection_error_px.toFixed(3)} px mean error · {statistics.elapsed_seconds.toFixed(1)} s solve</span>
      {frameQuality?.p90_error_px != null && <span>Frame {currentFrame.frame_number} · {frameQuality.observations} observations · {frameQuality.p90_error_px.toFixed(3)} px 90th-percentile error</span>}
      {currentFrame && solve.frame_quality && !frameQuality && <span>
        Frame {currentFrame.frame_number} · {currentSampled ? "Pose unavailable" : "Outside the solve sample"}
      </span>}
      <span>{solve.masking_applied ? "SAM exclusions applied" : "Baseline features used"}</span>
      {statistics.initial_focal_px != null && <span>{statistics.initial_focal_px} px initial focal length</span>}
      {solve.solver?.pairing && <span>{solve.solver.pairing.replaceAll("_", " ")} pairing · shared SIMPLE_RADIAL lens</span>}
      {solve.diagnostics?.geometry?.homography_fraction >= 0.5 && <span>Many pairs are explained by a homography. Inspect the 3D solve carefully.</span>}
      <CameraPath points={solve.camera_path ?? []} />
      <label className="check"><input type="checkbox" checked={showTracks} onChange={(e) => setShowTracks(e.target.checked)} /> Show reconstructed features</label>
      <button disabled={busy} onClick={() => onSolve(solve)}>Reload feature overlay</button>
      <div className="downloads">
        <a href={backend + solve.result_url}>Camera JSON</a><a href={backend + solve.tracks_url}>Tracks</a><a href={backend + solve.quality_url}>Quality</a>
        {solve.points_url && <a href={backend + solve.points_url}>Sparse PLY</a>}
      </div>
      {!sequence.fps && <label>Source FPS for export <input value={exportFps} placeholder="e.g. 24000/1001"
        disabled={busy} onChange={(e) => setExportFps(e.target.value)} /></label>}
      <button disabled={busy || (!sequence.fps && !exportFps.trim())} onClick={() => start("export", {
        source_job_id: solve.jobId, export_fps: sequence.fps ? null : exportFps.trim(),
      })}>Prepare Blender package</button>
      {exportUrl && <a href={backend + exportUrl}>Download Blender package</a>}
    </section>}
    {comparison && <section>
      <h2>Comparison</h2>
      {["baseline", "masked"].map((mode) => {
        const result = comparison[mode];
        const stats = result.statistics;
        return <div className="metadata" key={mode}>
          <strong>{mode === "baseline" ? "Baseline" : "Masked"}</strong>
          {result.status === "failed" ? <span>{result.error}</span> : <span>
            {stats.registered_frames}/{stats.input_frames} solved · {stats.mean_reprojection_error_px.toFixed(3)} px mean error
          </span>}
          <span>{stats.detected_keypoints?.toLocaleString() ?? "Unknown"} features · {stats.geometric_inlier_ratio == null ?
            "inlier ratio unavailable" : (100 * stats.geometric_inlier_ratio).toFixed(1) + "% geometric inliers"}
            {stats.elapsed_seconds != null && " · " + stats.elapsed_seconds.toFixed(1) + " s"}</span>
          {result.status !== "failed" && <button disabled={busy} onClick={() => start("solve", {
            comparison_job_id: comparison.jobId, comparison_mode: mode,
          })}>{mode === "baseline" ? "Open baseline result" : "Open masked result"}</button>}
        </div>;
      })}
      <span>{comparison.excluded_baseline_keypoints ?? "Unknown"} baseline keypoints in excluded regions</span>
      <a href={backend + comparison.comparison_url} target="_blank" rel="noreferrer">Full comparison</a>
    </section>}
  </aside>;
}
