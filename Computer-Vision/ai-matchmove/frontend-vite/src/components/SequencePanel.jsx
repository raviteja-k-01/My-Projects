import { useState, useRef } from "react";

export default function SequencePanel({ setFrames }) {
  const [loading, setLoading] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [showPopup, setShowPopup] = useState(false);
  const inputRef = useRef();

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    [...files].forEach((f) => formData.append("files", f));

    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      if (data.frames && data.frames.length > 0) {
        const sorted = data.frames.sort((a, b) =>
          a.localeCompare(b, undefined, { numeric: true })
        );

        setFrames(sorted);
        setFrameCount(sorted.length);

        // popup
        setShowPopup(true);
        setTimeout(() => setShowPopup(false), 2000);
      } else {
        alert("No frames returned");
      }
    } catch (err) {
      console.error(err);
      alert("Upload failed");
    }

    setLoading(false);

    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      style={{
        width: "180px",
        background: "#1a1a1a",
        borderRight: "1px solid #333",
        padding: "12px 10px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        position: "relative",
      }}
    >
      {/* LABEL */}
      <div
        style={{
          fontSize: "12px",
          fontWeight: "600",
          color: "#e6e6e6",
          letterSpacing: "1px",
        }}
      >
        SEQUENCE
      </div>

      {/* UPLOAD BUTTON */}
      <button
        onClick={() => inputRef.current.click()}
        style={{
          padding: "8px",
          background: "#2a2a2a",
          border: "1px solid #444",
          color: "#ddd",
          cursor: "pointer",
          fontSize: "12px",
          borderRadius: "3px",
        }}
      >
        {loading ? "Loading..." : "Upload"}
      </button>

      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".exr,image/*"
        style={{ display: "none" }}
        onChange={(e) => handleUpload(e.target.files)}
      />

      {/* POPUP */}
      {showPopup && (
        <div
          style={{
            position: "absolute",
            bottom: "12px",
            left: "10px",
            background: "#4da6ff",
            color: "black",
            padding: "6px 10px",
            fontSize: "11px",
            borderRadius: "3px",
          }}
        >
          Loaded ({frameCount})
        </div>
      )}
    </div>
  );
}