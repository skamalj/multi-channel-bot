/**
 * AG-UI passthrough to the FastAPI bridge.
 *
 * The browser could call the bridge directly, but same-origin keeps CORS out
 * of it and lets the bridge's address be configuration rather than something
 * baked into the client bundle.
 *
 * Nothing is parsed here. The body streams through byte for byte, because a
 * proxy that re-emitted the events would be a second implementation of the
 * protocol and the console would then be testing the proxy.
 */
export const dynamic = "force-dynamic";

const AGUI_URL = process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui";

export async function POST(req: Request) {
  const upstream = await fetch(AGUI_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: await req.text(),
    // @ts-expect-error - node fetch needs this to stream rather than buffer
    duplex: "half",
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      "x-accel-buffering": "no",
    },
  });
}
