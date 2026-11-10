"""
frontend/app.py — Multimodal AI Assistant UI
"""

import os
import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multimodal AI Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem; border-radius: 14px; color: white;
    text-align: center; margin-bottom: 1.5rem;
}
.header h1 { font-size: 2.2rem; margin: 0; }
.header p  { margin: 0.5rem 0 0; opacity: 0.75; font-size: 0.95rem; }
.chip {
    display: inline-block; border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; margin: 2px; font-weight: 600;
}
.doc   { background:#e3f2fd; color:#1565c0; }
.img   { background:#f3e5f5; color:#6a1b9a; }
.audio { background:#e8f5e9; color:#2e7d32; }
.video { background:#fff3e0; color:#e65100; }
.mbadge {
    display:inline-block; background:#1F6FEB22; color:#1F6FEB;
    border:1px solid #1F6FEB55; border-radius:20px;
    padding:2px 8px; font-size:0.7rem; margin-left:5px;
}
.tbadge {
    display:inline-block; background:#f3e5f5; color:#6a1b9a;
    border-radius:20px; padding:2px 8px; font-size:0.7rem; margin-left:4px;
}
.stButton>button {
    background:#1F6FEB; color:white; border:none;
    border-radius:8px; font-weight:600; width:100%;
}
.stButton>button:hover { background:#1558d6; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
for k, v in [("session_id", None), ("files", []), ("history", []), ("tokens", 0), ("model", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ──
st.markdown("""
<div class="header">
    <h1>🧠 Multimodal AI Assistant</h1>
    <p>PDF · Word · Excel · Images · Audio · Video — powered by <strong>Groq</strong> ⚡</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    fmt = st.selectbox("Output Format", ["auto", "table", "report", "bullets", "json"])

    st.markdown("---")
    st.markdown("### 📁 Supported Files")
    st.markdown("📄 PDF · 📝 DOCX · 📊 XLSX")
    st.markdown("🖼️ PNG · JPG · WEBP")
    st.markdown("🎵 MP3 · WAV · M4A · OGG")
    st.markdown("🎬 MP4 · AVI · MOV · MKV")

    st.markdown("---")
    st.markdown("### 🤖 Models (all free)")
    st.markdown("👁️ `llama-4-scout` — vision")
    st.markdown("📝 `llama-3.3-70b` — text")
    st.markdown("🎙️ `whisper-large-v3` — audio")

    if st.session_state.tokens > 0:
        st.markdown("---")
        st.metric("Tokens Used", f"{st.session_state.tokens:,}")
    if st.session_state.model:
        st.caption(f"Last: `{st.session_state.model.split('/')[-1]}`")

    st.markdown("---")
    if st.session_state.session_id:
        st.success("✅ Session active")
        if st.button("🗑️ Clear Session"):
            try:
                httpx.delete(f"{API_URL}/session/{st.session_state.session_id}")
            except Exception:
                pass
            st.session_state.update({"session_id": None, "files": [], "history": [], "tokens": 0, "model": ""})
            st.rerun()
    else:
        st.info("Upload files to begin.")

    st.markdown("---")
    st.markdown("**Ravi Teja Kesagani**")
    st.markdown("[GitHub](https://github.com/raviteja5) · [LinkedIn](https://linkedin.com/in/raviteja5)")

# ── Layout ──
left, right = st.columns([1, 1.6], gap="large")

# ── Upload ──
with left:
    st.markdown("### 📂 Upload Files")
    files = st.file_uploader(
        "Drop files",
        type=["pdf","docx","xlsx","xls","png","jpg","jpeg","webp","gif",
              "mp3","wav","m4a","ogg","flac","mp4","avi","mov","mkv","webm"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if files:
        if st.button("🚀 Process Files"):
            with st.spinner("Parsing... audio/video may take a moment ⏳"):
                try:
                    payload = [("files", (f.name, f.read(), f.type or "application/octet-stream")) for f in files]
                    r = httpx.post(f"{API_URL}/upload", files=payload, timeout=300.0)
                    if r.status_code == 200:
                        d = r.json()
                        st.session_state.session_id = d["session_id"]
                        st.session_state.files = d["files_uploaded"]
                        st.session_state.history = []
                        st.session_state.tokens = 0
                        st.success(f"✅ {len(d['files_uploaded'])} file(s) ready!")
                        if d.get("errors"):
                            for e in d["errors"]:
                                st.warning(f"⚠️ {e['filename']}: {e['error']}")
                    else:
                        st.error(f"Error: {r.text}")
                except httpx.ConnectError:
                    st.error("❌ Backend not running!\n\nRun:\n```\nuvicorn backend.main:app --reload --port 8000\n```")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.files:
        st.markdown("#### Loaded Files")
        chip_map = {"pdf":"doc","docx":"doc","xlsx":"doc","image":"img","audio":"audio","video":"video"}
        icon_map  = {"pdf":"📄","docx":"📝","xlsx":"📊","image":"🖼️","audio":"🎵","video":"🎬"}
        for f in st.session_state.files:
            ft = f.get("type","unknown") if isinstance(f, dict) else "unknown"
            fn = f.get("filename", f) if isinstance(f, dict) else f
            c  = chip_map.get(ft, "doc")
            ic = icon_map.get(ft, "📎")
            st.markdown(f'<span class="chip {c}">{ic} {fn}</span>', unsafe_allow_html=True)

    if st.session_state.session_id:
        st.markdown("#### 💡 Quick Prompts")
        for p in ["Summarize all content", "What are the key insights?",
                  "Transcribe the audio/video", "Compare all documents",
                  "Generate a full report", "List all data and figures"]:
            if st.button(p, key=f"q_{p}"):
                st.session_state["_pre"] = p
                st.rerun()

# ── Chat ──
with right:
    st.markdown("### 💬 Ask Anything")
    box = st.container(height=450)
    with box:
        if not st.session_state.history:
            st.markdown("""
            <div style='text-align:center;color:#888;padding:3rem 1rem;'>
                <div style='font-size:3rem'>🧠</div>
                <p>Upload any file on the left, then ask anything here.</p>
                <p style='font-size:0.82rem;color:#aaa;'>
                PDF · Word · Excel · Images · Audio · Video<br>
                Vision model auto-selected when needed ⚡
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            for turn in st.session_state.history:
                with st.chat_message("user"):
                    st.write(turn["prompt"])
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(turn["answer"])
                    badges = ""
                    if turn.get("model"):
                        m = turn["model"].split("/")[-1]
                        badges += f'<span class="mbadge">🤖 {m}</span>'
                    if turn.get("tokens"):
                        badges += f'<span class="tbadge">⚡ {turn["tokens"]:,} tok</span>'
                    if badges:
                        st.markdown(badges, unsafe_allow_html=True)

    pre = st.session_state.pop("_pre", "")
    prompt = st.chat_input("Ask a question, request a summary, analyse audio/video...",
                           disabled=not st.session_state.session_id)
    if pre and not prompt:
        prompt = pre

    if prompt:
        if not st.session_state.session_id:
            st.warning("Upload files first.")
        else:
            with st.spinner("⚡ Thinking..."):
                try:
                    r = httpx.post(
                        f"{API_URL}/ask",
                        json={"session_id": st.session_state.session_id,
                              "prompt": prompt, "output_format": fmt},
                        timeout=180.0
                    )
                    if r.status_code == 200:
                        d = r.json()
                        st.session_state.tokens += d.get("tokens_used", 0)
                        st.session_state.model = d.get("model_used", "")
                        st.session_state.history.append({
                            "prompt": prompt,
                            "answer": d["answer"],
                            "tokens": d.get("tokens_used", 0),
                            "model":  d.get("model_used", "")
                        })
                        st.rerun()
                    else:
                        st.error(f"API error: {r.text}")
                except httpx.ConnectError:
                    st.error("❌ Backend not running.")
                except Exception as e:
                    st.error(f"Error: {e}")
