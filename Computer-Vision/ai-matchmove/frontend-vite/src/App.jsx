import { useState, useEffect, useRef } from "react";
import SequencePanel from "./components/SequencePanel";
import Viewer from "./components/Viewer";
import Timeline from "./components/Timeline";
import { previewUrl, request } from "./api";

export default function App() {
  const [sequence, setSequence] = useState(null);
  const [projects, setProjects] = useState([]);
  const [projectError, setProjectError] = useState("");
  const [shownIndex, setShownIndex] = useState(null);
  const [requested, setRequested] = useState(null);
  const [playbackError, setPlaybackError] = useState("");
  const currentIndex = shownIndex ?? 0;
  const [isPlaying, setIsPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [prompts, setPrompts] = useState([]);
  const [tool, setTool] = useState("none");
  const [manifest, setManifest] = useState(null);
  const [tracks, setTracks] = useState({});
  const [tracksError, setTracksError] = useState("");
  const [showTracks, setShowTracks] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const activeSequence = useRef(null);
  const tracksRequest = useRef(null);
  const lastFrameDeadline = useRef(null);
  const previews = useRef(new Map());
  const frames = sequence?.frame_details ?? [];

  const handleSequence = (nextSequence) => {
    tracksRequest.current?.abort();
    tracksRequest.current = null;
    activeSequence.current = nextSequence.sequence_id;
    sessionStorage.setItem("matchmove.sequence", nextSequence.sequence_id);
    localStorage.setItem("matchmove.sequence", nextSequence.sequence_id);
    setProjects((items) => [{ sequence_id: nextSequence.sequence_id,
      source_name: nextSequence.source_name ?? nextSequence.frame_details[0]?.source_name ?? "Image sequence",
      frame_count: nextSequence.frames.length }, ...items.filter((item) => item.sequence_id !== nextSequence.sequence_id)]);
    setSequence(nextSequence);
    setShownIndex(null);
    previews.current = new Map();
    const savedFrame = localStorage.getItem("matchmove.frame." + nextSequence.sequence_id);
    const index = nextSequence.frame_details.findIndex((item) => String(item.frame_number) === savedFrame);
    setRequested(nextSequence.frames.length ? { index: Math.max(index, 0) } : null);
    setPlaybackError("");
    setIsPlaying(false);
    setPrompts([]);
    setManifest(null);
    setTracks({});
    setTracksError("");
    setTool("none");
  };

  useEffect(() => {
    const id = localStorage.getItem("matchmove.sequence") ?? sessionStorage.getItem("matchmove.sequence");
    let stale = false;
    request("/sequences").then((saved) => {
      if (!stale) setProjects(saved.sequences);
    }).catch((error) => { if (!stale) setProjectError(error.message); });
    if (id) request("/sequences/" + encodeURIComponent(id)).then((saved) => {
      if (!stale && !activeSequence.current) handleSequence(saved);
    }).catch(() => {
      if (!stale && !activeSequence.current) {
        sessionStorage.removeItem("matchmove.sequence");
        localStorage.removeItem("matchmove.sequence");
      }
    });
    return () => { stale = true; };
  }, []);

  const seek = (index) => {
    lastFrameDeadline.current = null;
    setIsPlaying(false);
    setRequested({ index: typeof index === "function" ? index(currentIndex) : index });
  };

  function loadFrame(frame) {
    const cache = previews.current;
    const url = previewUrl(frame);
    if (!cache.has(url)) {
      const image = new Image();
      image.src = url;
      cache.set(url, image.decode().then(() => image).catch((error) => {
        cache.delete(url);
        throw error;
      }));
      if (cache.size > 12) cache.delete(cache.keys().next().value);
    }
    return cache.get(url);
  }

  useEffect(() => {
    if (!requested || !sequence?.frame_details[requested.index]) return;
    let stale = false;
    setPlaybackError("");
    const timer = setTimeout(() => {
      loadFrame(sequence.frame_details[requested.index]).then(() => {
        if (stale) return;
        setShownIndex(requested.index);
        setRequested(null);
      }).catch(() => {
        if (!stale) {
          setRequested(null);
          setIsPlaying(false);
          setPlaybackError("Could not load the selected frame preview.");
        }
      });
    }, 45);
    return () => { stale = true; clearTimeout(timer); };
  }, [requested, sequence]);

  useEffect(() => {
    if (!isPlaying || requested || shownIndex === null || !sequence?.frame_details.length) {
      lastFrameDeadline.current = null;
      return;
    }
    const frames = sequence.frame_details;
    const nextIndex = shownIndex + 1;
    if (nextIndex >= frames.length) {
      setIsPlaying(false);
      return;
    }
    const current = frames[shownIndex];
    const next = frames[nextIndex];
    const frameInterval = 1000 / (sequence.fps ?? 24);
    const currentTime = current.timestamp_seconds;
    const nextTime = next.timestamp_seconds;
    const interval = Number.isFinite(currentTime) && Number.isFinite(nextTime) && nextTime > currentTime
      ? (nextTime - currentTime) * 1000 : (next.frame_number - current.frame_number) * frameInterval;
    const deadline = (lastFrameDeadline.current ?? performance.now()) + interval;
    let stale = false;
    let timer;
    const wait = (done) => {
      const remaining = deadline - performance.now();
      if (remaining <= 0) {
        done();
      } else {
        timer = setTimeout(() => wait(done), Math.min(remaining, 2_147_483_647));
      }
    };
    const nextReady = loadFrame(next);
    for (const frame of frames.slice(nextIndex + 1, nextIndex + 5)) loadFrame(frame).catch(() => {});
    Promise.all([nextReady, new Promise(wait)]).then(() => {
      if (stale) return;
      const now = performance.now();
      lastFrameDeadline.current = now - deadline > frameInterval ? now : deadline;
      setShownIndex(nextIndex);
    }).catch(() => {
      if (!stale) {
        setPlaybackError("Playback paused because the next preview could not load.");
        setIsPlaying(false);
      }
    });
    return () => { stale = true; clearTimeout(timer); };
  }, [isPlaying, sequence, shownIndex, requested]);

  const currentFrame = shownIndex === null ? null : frames[shownIndex];
  useEffect(() => {
    if (!isPlaying && currentFrame) {
      localStorage.setItem("matchmove.frame." + sequence.sequence_id, currentFrame.frame_number);
    }
  }, [currentFrame, isPlaying, sequence?.sequence_id]);
  const prompt = prompts.find((item) => item.frame_number === currentFrame?.frame_number);
  const mask = manifest?.frames.find((item) => item.frame_number === currentFrame?.frame_number);

  function updatePrompt(change) {
    if (!currentFrame) return;
    setIsPlaying(false);
    const others = prompts.filter((item) => item.frame_number !== currentFrame.frame_number);
    if (!change) {
      setPrompts(others);
      return;
    }
    const previous = prompt ?? { frame_number: currentFrame.frame_number, points: [], decision: "auto" };
    const { point, ...fields } = change;
    setPrompts([...others, { ...previous, ...fields,
      points: fields.points ?? (point ? [...previous.points, point] : previous.points),
    }]);
  }

  async function loadTracks(result) {
    tracksRequest.current?.abort();
    const controller = new AbortController();
    tracksRequest.current = controller;
    setTracks({});
    setTracksError("");
    try {
      const data = await request(result.tracks_url, { signal: controller.signal });
      if (tracksRequest.current === controller) {
        setTracks(data.frames);
        setShowTracks(true);
      }
    } catch {
      if (!controller.signal.aborted && tracksRequest.current === controller) {
        setTracksError("Could not load the feature overlay. Reopen this solve to try again.");
      }
    }
  }

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        background: "#1a1a1a",
        color: "white",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "34px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "rgba(0,0,0,0.35)",
          
          zIndex: 10,
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            fontSize: "14px",
            fontWeight: "700",
            color: "#ffffff",
            letterSpacing: "1.4px",
          }}
        >
          AI MATCHMOVE ENGINE
        </div>
      </div>

      <SequencePanel key={sequence?.sequence_id ?? "new"} onSequence={handleSequence} sequence={sequence} currentFrame={currentFrame}
        projects={projects} projectError={projectError} onOpenProject={async (id) => {
          setBusy(true);
          setProjectError("");
          try { handleSequence(await request("/sequences/" + encodeURIComponent(id))); }
          catch (error) { setProjectError(error.message); }
          finally { setBusy(false); }
        }} onDiscardPrompts={() => setPrompts([])}
        prompts={prompts} tool={tool} setTool={(value) => { setIsPlaying(false); setTool(value); }}
        onPrompt={updatePrompt} manifest={manifest} setManifest={setManifest}
        onSolve={loadTracks} busy={busy} setBusy={setBusy}
        showTracks={showTracks} setShowTracks={setShowTracks} showMask={showMask} setShowMask={setShowMask}
        onSelectSourceFrame={(number) => {
          const index = frames.findIndex((frame) => frame.frame_number === number);
          if (index !== -1) seek(index);
        }} />

      <div
        style={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Viewer frame={currentFrame} preview={showMask ? mask?.preview_url : null}
          sampleAspect={sequence?.video_metadata?.sample_aspect_ratio}
          tracks={showTracks ? tracks[currentFrame?.frame_number] : null}
          prompt={prompt} tool={busy ? "none" : tool} onPrompt={updatePrompt} />

        <Timeline
          frames={frames}
          currentIndex={currentIndex}
          setCurrentIndex={seek}
          isPlaying={isPlaying}
          setIsPlaying={(play) => {
            if (play && shownIndex === frames.length - 1) setRequested({ index: 0 });
            setPlaybackError("");
            setIsPlaying(play);
          }}
        />
        {playbackError && <div role="alert" style={{ padding: 6, color: "#ff9a92" }}>{playbackError}</div>}
        {tracksError && <div role="alert" style={{ padding: 6, color: "#ff9a92" }}>{tracksError}</div>}
      </div>
    </div>
  );
}
