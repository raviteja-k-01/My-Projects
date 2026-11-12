import { useLayoutEffect, useRef } from "react";

export default function Timeline({
  frames,
  currentIndex,
  setCurrentIndex,
  isPlaying,
  setIsPlaying,
}) {
  const strip = useRef(null);
  const active = useRef(null);
  useLayoutEffect(() => {
    if (!strip.current || !active.current) return;
    const container = strip.current.getBoundingClientRect();
    const item = active.current.getBoundingClientRect();
    if (item.left < container.left) strip.current.scrollLeft += item.left - container.left;
    else if (item.right > container.right) strip.current.scrollLeft += item.right - container.right;
  }, [currentIndex, frames]);
  const btn = {
    padding: "6px 10px",
    background: "#222",
    border: "1px solid #444",
    color: "white",
    cursor: "pointer",
  };

  const btnMain = {
    ...btn,
    background: "#4da6ff",
    color: "black",
    fontWeight: "bold",
  };

  return (
    <div
      style={{
        background: "#111",
        borderTop: "1px solid #333",
        height: "110px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        padding: "8px 16px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
        }}
      >
        <button
          aria-label="First frame"
          onClick={() => setCurrentIndex(0)}
          style={btn}
          disabled={!frames.length}
        >
          ⏮
        </button>

        <button
          aria-label="Previous frame"
          onClick={() =>
            setCurrentIndex((prev) => Math.max(prev - 1, 0))
          }
          style={btn}
          disabled={!frames.length}
        >
          ◀
        </button>

        <button
          aria-label={isPlaying ? "Pause" : "Play"}
          onClick={() => setIsPlaying(!isPlaying)}
          style={btnMain}
          disabled={!frames.length}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        <button
          aria-label="Next frame"
          onClick={() =>
            setCurrentIndex((prev) =>
              Math.min(prev + 1, frames.length - 1)
            )
          }
          style={btn}
          disabled={!frames.length}
        >
          ▶
        </button>

        <button
          aria-label="Last frame"
          onClick={() => setCurrentIndex(frames.length - 1)}
          style={btn}
          disabled={!frames.length}
        >
          ⏭
        </button>

        <div
          style={{
            marginLeft: 20,
            color: "#aaa",
            fontSize: 13,
            minWidth: 120,
          }}
        >
          Frame: {frames[currentIndex]?.frame_number ?? "–"}
          {frames.length > 0 && ` (${currentIndex + 1} / ${frames.length})`}
        </div>
      </div>

      <div
        ref={strip}
        style={{
          display: "flex",
          overflowX: "auto",
          gap: 4,
          paddingTop: 6,
        }}
      >
        {frames.map((frame, i) => (
          <button
            ref={i === currentIndex ? active : null}
            aria-current={i === currentIndex ? "true" : undefined}
            aria-label={`Go to frame ${frame.frame_number}`}
            key={frame.url}
            onClick={() => setCurrentIndex(i)}
            style={{
              minWidth: 32,
              height: 42,
              background: i === currentIndex ? "#4da6ff" : "#333",
              border:
                i === currentIndex
                  ? "2px solid orange"
                  : "1px solid #555",
              cursor: "pointer",
              textAlign: "center",
              lineHeight: "42px",
              fontSize: 11,
              color: "white",
              userSelect: "none",
              flexShrink: 0,
            }}
          >
            {frame.frame_number}
          </button>
        ))}
      </div>
    </div>
  );
}
