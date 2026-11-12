export const backend = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export function previewUrl(frame) {
  return backend + frame.url.replace(/^\/frames\//, "/previews/");
}

export async function request(path, options) {
  const response = await fetch(backend + path, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (typeof data.detail === "string") throw new Error(data.detail);
    if (Array.isArray(data.detail)) {
      const issues = data.detail.map(({ loc, msg }) =>
        [loc?.at(-1)?.toString().replaceAll("_", " "), msg?.replace(/^Value error, /, "")].filter(Boolean).join(" · "));
      throw new Error(issues.join("; "));
    }
    throw new Error("Request failed (" + response.status + ").");
  }
  return response.json();
}
