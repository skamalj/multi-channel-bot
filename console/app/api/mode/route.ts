/** Which agent is answering: the deployed Runtime, or this laptop. */
export const dynamic = "force-dynamic";

export async function GET() {
  const base = (process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui")
    .replace(/\/agui$/, "");
  try {
    const r = await fetch(`${base}/api/agui/mode`, { cache: "no-store" });
    return Response.json(await r.json());
  } catch {
    return Response.json({ mode: "local", runtime_arn: null });
  }
}
