export default function Timeline({
  frames,
  currentIndex,
  setCurrentIndex,
  isPlaying,
  setIsPlaying,
}) {
  // BUTTON STYLES
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
      {/* CONTROLS */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
        }}
      >
        {/* FIRST FRAME */}
        <button
          onClick={() => setCurrentIndex(0)}
          style={btn}
          disabled={!frames.length}
        >
          ⏮
        </button>

        {/* PREV FRAME */}
        <button
          onClick={() =>
            setCurrentIndex((prev) => Math.max(prev - 1, 0))
          }
          style={btn}
          disabled={!frames.length}
        >
          ◀
        </button>

        {/* PLAY / PAUSE */}
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          style={btnMain}
          disabled={!frames.length}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        {/* NEXT FRAME */}
        <button
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

        {/* LAST FRAME */}
        <button
          onClick={() => setCurrentIndex(frames.length - 1)}
          style={btn}
          disabled={!frames.length}
        >
          ⏭
        </button>

        {/* FRAME INFO */}
        <div
          style={{
            marginLeft: 20,
            color: "#aaa",
            fontSize: 13,
            minWidth: 120,
          }}
        >
          Frame: {frames.length ? currentIndex + 1 : 0} / {frames.length}
        </div>
      </div>

      {/* FRAME STRIP */}
      <div
        style={{
          display: "flex",
          overflowX: "auto",
          gap: 4,
          paddingTop: 6,
        }}
      >
        {frames.map((_, i) => (
          <div
            key={i}
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
              transition: "0.1s",
              userSelect: "none",
            }}
          >
            {i}
          </div>
        ))}
      </div>
    </div>
  );
}