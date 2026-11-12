import { spawn } from "node:child_process";
import { once } from "node:events";
import { access, mkdtemp } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

const restore = process.argv[2] === "--restore";
const analyze = process.argv[2] === "--analyze-review";
const review = analyze || process.argv[2] === "--review" ? process.argv[3] : null;
const exportOnly = process.argv[2] === "--export-only" ? process.argv[3] : null;
const compareExport = process.argv[2] === "--compare-export" ? process.argv[3] : null;
const demo = process.argv[2] === "--demo";
const backend = process.env.MATCHMOVE_E2E_BACKEND ?? "http://127.0.0.1:8027";
if (!process.argv[2]) throw new Error("Pass the real video path, --demo <folder>, --restore, --review, --analyze-review, --export-only, or --compare-export.");
if (demo && !process.argv[3]) throw new Error("Pass the generated image folder after --demo.");
const footage = demo
  ? Array.from({ length: 9 }, (_, i) => resolve(process.argv[3], `synthetic_${String(101 + i).padStart(4, "0")}.png`))
  : [resolve(process.argv[2])];
if (demo) await Promise.all(footage.map((file) => access(file)));
const pause = (ms) => new Promise((done) => setTimeout(done, ms));
async function wait(check, label, timeout = 120000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const result = await check().catch(() => false);
    if (result) return result;
    await pause(300);
  }
  throw new Error("Timed out: " + label);
}
const ui = spawn(process.execPath, ["node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", "5187", "--strictPort"], {
  env: { ...process.env, VITE_API_URL: backend }, windowsHide: true, stdio: "ignore",
});
const profile = process.env.MATCHMOVE_CHROME_PROFILE ?? await mkdtemp(join(tmpdir(), "matchmove-chrome-"));
console.log("Chrome profile", profile);
const chrome = spawn(process.env.MATCHMOVE_CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe", [
  "--headless=new", "--disable-gpu", "--remote-debugging-port=9237", "--user-data-dir=" + profile,
  "--no-first-run", "about:blank",
], { windowsHide: true, stdio: "ignore" });
let ws;
try {
  await wait(async () => (await fetch("http://127.0.0.1:5187")).ok, "Vite", 12000);
  const tab = await wait(async () => (await (await fetch("http://127.0.0.1:9237/json")).json()).find(t => t.type === "page"), "Chrome", 12000);
  ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((done, reject) => { ws.addEventListener("open", done, { once: true }); ws.addEventListener("error", reject, { once: true }); });
  let id = 0;
  const inflight = new Map();
  const exceptions = [];
  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.method === "Runtime.exceptionThrown") exceptions.push(data.params.exceptionDetails.text);
    if (!inflight.has(data.id)) return;
    const [done, reject] = inflight.get(data.id);
    inflight.delete(data.id);
    if (data.error) reject(new Error(data.error.message)); else done(data.result);
  });
  function cdp(method, params = {}) {
    const key = ++id;
    return new Promise((done, reject) => {
      inflight.set(key, [done, reject]);
      ws.send(JSON.stringify({ id: key, method, params }));
    });
  }
  async function evaluate(expression) {
    const data = await cdp("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (data.exceptionDetails) throw new Error(data.exceptionDetails.exception?.description ?? data.exceptionDetails.text);
    return data.result.value;
  }
  const text = () => evaluate("document.body.innerText");
  const click = (label) => evaluate("Array.from(document.querySelectorAll('button')).find(b=>b.textContent===" + JSON.stringify(label) + ").click()");
  const setInput = (label, value) => evaluate(`(() => {
    const input = [...document.querySelectorAll("label")].find(label => label.textContent.startsWith(${JSON.stringify(label)})).querySelector("input");
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, ${JSON.stringify(value)});
    input.dispatchEvent(new Event("input", {bubbles:true}));
  })()`);
  const openSaved = (id) => wait(() => evaluate(`(() => {
    const picker = document.querySelector("select");
    const id = ${JSON.stringify(id)};
    if (![...picker.options].some(option => option.value === id)) return false;
    picker.value = id;
    picker.dispatchEvent(new Event("change", {bubbles:true}));
    return true;
  })()`), "saved footage");
  async function downloadExport() {
    const link = await wait(() => evaluate("Array.from(document.querySelectorAll('a')).find(a=>a.textContent==='Download Blender package')?.href"), "Blender package");
    const response = await fetch(link);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (!response.ok || bytes[0] !== 80 || bytes[1] !== 75) throw new Error("Blender package download failed");
    return { link, exportBytes: bytes.length };
  }
  await cdp("Runtime.enable");
  await cdp("Page.navigate", { url: "http://127.0.0.1:5187" });
  await wait(async () => (await text()).includes("Upload video or sequence"), "React app");
  if (restore) {
    const frame = Number(process.argv[3] ?? 8);
    await wait(() => evaluate(`document.querySelector('img')?.alt === 'Source frame ${frame}'`), "restored footage");
    await wait(async () => (await text()).includes("18/18 sampled frames solved"), "restored solve");
    await wait(async () => (await text()).includes("Download Blender package"), "restored export");
    if (frame !== 8) {
      await evaluate("document.querySelector('[aria-label=\"Go to frame 8\"]').click()");
      await wait(() => evaluate("document.querySelector('img')?.alt === 'Source frame 8'"), "seek before closing");
    }
    console.log("A fresh Chrome process restored footage, frame, reconstruction and export.");
  } else if (review) {
    await openSaved(review);
    if (analyze) {
      await wait(() => evaluate("Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Generate and review masks'&&!b.disabled)"), "foreground controls");
      const jobHistory = `${backend}/sequences/${review}/jobs`;
      const before = (await (await fetch(jobHistory)).json()).jobs[0]?.job_id;
      await setInput("SAM analysis frames", 5);
      await click("Generate and review masks");
      const realSam = await wait(async () => {
        const { jobs } = await (await fetch(jobHistory)).json();
        return jobs.find(job => job.job_id !== before && job.status === "done" && job.result?.model === "SAM ViT-B");
      }, "real SAM ViT-B job from Chrome");
      if (realSam.result.analyzed_frames !== 5) throw new Error("Browser analysis settings were not used");
    }
    await wait(() => evaluate("Array.from(document.querySelectorAll('button')).some(b => b.textContent.startsWith('Next candidate frame') && !b.disabled)"), "saved automatic foreground candidates");
    await evaluate("Array.from(document.querySelectorAll('button')).find(b=>b.textContent.startsWith('Next candidate frame')).click()");
    await wait(() => evaluate("document.querySelector('img[alt=\"Foreground mask preview\"]')?.naturalWidth > 0"), "real SAM mask preview");
    const revisions = `${backend}/sequences/${review}/foreground`;
    const before = await (await fetch(revisions)).json();
    await click("Keep region");
    await wait(async () => (await text()).includes("manual keep (reviewed)"), "saved manual correction");
    const after = await (await fetch(revisions)).json();
    if (after.mask_sets.length !== before.mask_sets.length + 1) throw new Error("Mask review did not save a revision");
    await cdp("Page.reload");
    await wait(async () => (await text()).includes("manual keep (reviewed)"), "restored mask correction");
    if (!analyze) await wait(async () => (await text()).includes("Download solve diagnostics"), "restored unsuitable-geometry failure");
    console.log(analyze ? "Chrome generated, previewed, corrected and restored real SAM masks." :
      "Chrome restored a real SAM mask correction and the fixed-camera solve failure.");
  } else if (exportOnly) {
    await openSaved(exportOnly);
    await wait(() => evaluate("document.body.innerText.includes('Source FPS for export')"), "missing source FPS export field");
    await setInput("Source FPS for export", "24000/1001");
    await click("Prepare Blender package");
    console.log(await downloadExport());
  } else if (compareExport) {
    await openSaved(compareExport);
    await wait(() => evaluate("Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Open masked result'&&!b.disabled)"), "saved comparison");
    await click("Open masked result");
    await wait(async () => (await text()).includes("SAM exclusions applied"), "saved masked camera solve");
    await wait(async () => (await text()).includes("reconstructed observations"), "masked camera observations");
    const history = await (await fetch(`${backend}/sequences/${compareExport}/jobs`)).json();
    const opened = history.jobs.find(item => item.operation === "solve" && item.result?.comparison_mode === "masked");
    if (!opened?.result.comparison_job_id) throw new Error("The opened solve did not reference the saved comparison");
    await click("Prepare Blender package");
    const exported = await downloadExport();
    await cdp("Page.reload");
    await wait(async () => (await text()).includes("Download Blender package"), "restored masked package");
    console.log({ comparedSolve: opened.job_id, ...exported });
  } else {
    const root = (await cdp("DOM.getDocument")).root.nodeId;
    const field = (await cdp("DOM.querySelector", { nodeId: root, selector: 'input[type=file]' })).nodeId;
    const clickControl = (name) => evaluate(`document.querySelector('[aria-label=' + ${JSON.stringify(JSON.stringify(name))} + ']').click()`);
    const firstFrame = demo ? 101 : 0;
    const nextFrame = demo ? 102 : 8;
    const frameCount = demo ? 9 : 18;
    const isShowing = (frame) => evaluate(`document.querySelector('img')?.alt === 'Source frame ${frame}'`);
    if (demo) await setInput("Sequence FPS", "24000/1001");
    else await setInput("Video sampling step", 8);
    await cdp("DOM.setFileInputFiles", { nodeId: field, files: footage });
    const imported = demo ? "9 available frames" : "18 available / 139 source frames";
    await wait(async () => (await text()).includes(imported), "imported footage");
    await wait(() => evaluate("document.querySelector('img')?.naturalWidth > 0"), "rendered footage");
    await clickControl("Play");
    await wait(async () => !(await text()).includes(`Frame: ${firstFrame} (1 / ${frameCount})`), "timeline playback");
    await clickControl("First frame");
    await wait(() => isShowing(firstFrame), "timeline seek");
    await click("Start Camera Solve");
    if (demo) {
      await wait(() => evaluate("Boolean(document.querySelector('[role=status] progress'))"), "solve progress", 30000);
    } else {
      await wait(async () => (await text()).includes("Extracting SIFT features"), "native solve progress", 30000);
      await click("Cancel");
      await wait(async () => (await text()).includes("Job cancelled."), "cancelled native solve", 30000);
      await click("Retry solve");
    }
    const solved = demo ? "9/9 sampled frames solved" : "sampled frames solved";
    await wait(async () => (await text()).includes(solved), "camera reconstruction", 180000);
    await wait(async () => /\b[1-9]\d* reconstructed observations/.test(await text()), "feature overlay");
    const snapshot = (await text()).split("\n").filter(line => /sampled frames solved|sparse points|mean error|90th-percentile/.test(line));
    if (!snapshot.some(line => line.includes("90th-percentile"))) throw new Error("Missing frame-specific reprojection statistics");
    await click("Prepare Blender package");
    const exported = await downloadExport();
    await clickControl("Next frame");
    await wait(() => isShowing(nextFrame), "timeline resume position");
    await cdp("Page.reload");
    await wait(() => isShowing(nextFrame), "restored frame");
    await wait(async () => (await text()).includes(solved), "restored reconstruction");
    await wait(async () => (await text()).includes("Download Blender package"), "restored export");
    console.log({snapshot, ...exported, recoveredFrame: nextFrame});
  }
  if (exceptions.length) throw new Error(JSON.stringify(exceptions));
} catch (error) {
  console.error(error);
  process.exitCode = 1;
} finally {
  if (ws?.readyState === WebSocket.OPEN) {
    const exited = once(chrome, "exit");
    ws.send(JSON.stringify({ id: 1000000, method: "Browser.close" }));
    await Promise.race([exited, pause(5000)]);
  }
  ws?.close();
  if (chrome.exitCode === null && chrome.signalCode === null) chrome.kill();
  ui.kill();
}
