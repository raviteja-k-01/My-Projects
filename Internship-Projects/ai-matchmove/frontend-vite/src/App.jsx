import { useState, useEffect } from "react";
import SequencePanel from "./components/SequencePanel";
import Viewer from "./components/Viewer";
import Timeline from "./components/Timeline";

export default function App() {
  const [frames, setFrames] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // ---------------------------
  // HANDLE NEW FRAMES
  // ---------------------------
  const handleFrames = (newFrames) => {
    setFrames(newFrames);
    setCurrentIndex(0);
    setIsPlaying(false);
  };

  // ---------------------------
  // PLAYBACK LOOP
  // ---------------------------
  useEffect(() => {
    if (!isPlaying || frames.length === 0) return;

    const interval = setInterval(() => {
      setCurrentIndex((prev) =>
        prev + 1 < frames.length ? prev + 1 : 0
      );
    }, 1000 / 24);

    return () => clearInterval(interval);
  }, [isPlaying, frames]);

  // ---------------------------
  // CURRENT FRAME
  // ---------------------------
  const currentFrame =
    frames.length > 0 ? frames[currentIndex] : null;

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
      {/* TOP OVERLAY STRIP (PRO LOOK) */}
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

      {/* LEFT PANEL */}
      <SequencePanel setFrames={handleFrames} />

      {/* MAIN AREA */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* VIEWER */}
        <Viewer frame={currentFrame} />

        {/* TIMELINE */}
        <Timeline
          frames={frames}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
        />
      </div>
    </div>
  );
}