/** Proxy to the bridge's trace store, so the page stays same-origin. */
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  ctx: { params: Promise<{ thread: string }> },
) {
  const { thread } = await ctx.params;
  const base = (process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui")
    .replace(/\/agui$/, "");
  try {
    const r = await fetch(`${base}/api/agui/trace/${encodeURIComponent(thread)}`,
      { cache: "no-store" });
    return Response.json(await r.json());
  } catch {
    return Response.json({ thread, events: [] });
  }
}
