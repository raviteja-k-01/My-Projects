import { useEffect, useState } from "react";

export default function Viewer({ frame }) {
  const [imgSrc, setImgSrc] = useState(null);

  const frameUrl = frame
    ? `http://127.0.0.1:8000/frames/${frame}`
    : null;

  // PRELOAD IMAGE (smooth playback)
  useEffect(() => {
    if (!frameUrl) return;

    const img = new Image();
    img.src = frameUrl;

    img.onload = () => {
      setImgSrc(frameUrl);
    };
  }, [frameUrl]);

  return (
    <div
      style={{
        flex: 1,
        background: "#111",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        overflow: "hidden",
      }}
    >
      {imgSrc ? (
        <img
          src={imgSrc}
          alt=""
          style={{
            width: "95%",
            height: "87%",
            objectFit: "contain", // ✅ keeps aspect ratio
          }}
        />
      ) : (
        <div style={{ color: "#555" }}>No Frame Loaded</div>
      )}
    </div>
  );
}