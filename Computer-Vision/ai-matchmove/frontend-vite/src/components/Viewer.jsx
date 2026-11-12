import { useState } from "react";
import { backend, previewUrl } from "../api";

export default function Viewer({ frame, preview, tracks, prompt, tool, onPrompt, sampleAspect }) {
  const [drag, setDrag] = useState(null);
  if (!frame) return <div style={{ flex: 1, display: "grid", placeItems: "center", background: "#111" }}>Upload footage to begin</div>;
  const { width, height } = frame;
  const [sampleX, sampleY] = (sampleAspect ?? "1:1").split(":").map(Number);
  const displayWidth = sampleX > 0 && sampleY > 0 ? width * sampleX / sampleY : width;
  const currentDrag = drag?.frame === frame.frame_number ? drag : null;
  const box = currentDrag ? [
    Math.min(currentDrag.start[0], currentDrag.end[0]), Math.min(currentDrag.start[1], currentDrag.end[1]),
    Math.max(currentDrag.start[0], currentDrag.end[0]), Math.max(currentDrag.start[1], currentDrag.end[1]),
  ] : prompt?.box;

  function coordinate(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [
      Math.min(width - 0.01, Math.max(0, (event.clientX - bounds.left) * width / bounds.width)),
      Math.min(height - 0.01, Math.max(0, (event.clientY - bounds.top) * height / bounds.height)),
    ];
  }

  function down(event) {
    if (event.button !== 0 || tool === "none") return;
    const xy = coordinate(event);
    if (tool !== "box") {
      onPrompt({ point: [...xy, tool === "foreground" ? 1 : 0] });
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ frame: frame.frame_number, start: xy, end: xy });
  }

  function up(event) {
    if (!currentDrag) return;
    const [x, y] = coordinate(event);
    const [sx, sy] = currentDrag.start;
    setDrag(null);
    if (Math.abs(x - sx) >= 2 && Math.abs(y - sy) >= 2) {
      onPrompt({ box: [Math.min(x, sx), Math.min(y, sy), Math.max(x, sx), Math.max(y, sy)] });
    }
  }

  const pointRadius = Math.max(width, height) / 330;
  const trackStep = Math.max(1, Math.ceil((tracks?.length ?? 0) / 400));
  return <div style={{ flex: 1, minHeight: 0, padding: "36px 10px 10px", background: "#111" }}>
    <div style={{ height: "100%", containerType: "size", display: "grid", placeItems: "center" }}>
      <div style={{ position: "relative", width: "min(100cqw, " + (100 * displayWidth / height) + "cqh)", aspectRatio: displayWidth + " / " + height }}>
        <img src={previewUrl(frame)} alt={"Source frame " + frame.frame_number}
          style={{ position: "absolute", width: "100%", height: "100%", objectFit: "fill" }} />
        {preview && <img key={preview} src={backend + preview} alt="Foreground mask preview"
          style={{ position: "absolute", width: "100%", height: "100%", objectFit: "fill" }} />}
        <svg viewBox={"0 0 " + width + " " + height} preserveAspectRatio="none"
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", cursor: tool === "none" ? "default" : "crosshair", touchAction: "none" }}
          onPointerDown={down} onPointerUp={up} onPointerCancel={() => setDrag(null)}
          onPointerMove={(event) => { if (currentDrag) setDrag({ ...currentDrag, end: coordinate(event) }); }}>
          {tracks?.filter((_, i) => i % trackStep === 0).map(([x, y, id]) =>
            <circle key={id} cx={x} cy={y} r={pointRadius * .55} fill="#66edbc" opacity=".8"><title>3D point {id}</title></circle>)}
          {box && <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill="none" stroke="#62b5f8" strokeWidth={pointRadius} />}
          {prompt?.points.map(([x, y, label], i) =>
            <circle key={i} cx={x} cy={y} r={pointRadius * 1.3} fill={label ? "#79ffaa" : "#ff807b"} stroke="#191919" />)}
        </svg>
        {tracks && <span style={{ position: "absolute", bottom: 5, left: 7, padding: 4, background: "#111c", fontSize: 11, pointerEvents: "none" }}>{tracks.length} reconstructed observations</span>}
      </div>
    </div>
  </div>;
}
