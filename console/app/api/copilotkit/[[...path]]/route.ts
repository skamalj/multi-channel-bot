/**
 * The CopilotKit runtime.
 *
 * AG-UI has two sides. The AGENT emits the events - that is `app/agui.py` and
 * the AgentCore Runtime container. The CLIENT consumes them, and this is the
 * hinge: the runtime wraps our agent as an `HttpAgent`, and the React app
 * talks to the runtime.
 *
 * It points at the FastAPI bridge rather than at AgentCore directly for a
 * reason that cannot be designed away: invoking the Runtime needs SigV4, and
 * neither a browser nor this Node process should hold AWS credentials. The
 * bridge signs, and streams the response through untouched.
 *
 *   browser -> /api/copilotkit -> HttpAgent -> FastAPI /agui -> AgentCore
 *                                                  (SigV4)
 *
 * The bridge also runs the agent in-process when no runtime ARN is set, so
 * this console drives a laptop or the deployment with nothing changed here.
 *
 * TWO THINGS THAT COST AN HOUR, both worth stating plainly:
 *
 * 1. This must be a CATCH-ALL route. The runtime serves several paths under
 *    its basePath - `/info`, which is how agents are discovered, among them.
 *    A plain `route.ts` answers only the base path, every sub-path falls
 *    through to Next's 404 page, discovery finds nothing, and the chat has no
 *    agent to talk to while looking perfectly healthy.
 *
 * 2. The v1 helper (`copilotRuntimeNextJSAppRouterEndpoint`) does not serve
 *    the v2 provider. `@copilotkit/react-core/v2` speaks to
 *    `@copilotkit/runtime/v2`, and mixing them yields "Method not allowed"
 *    from an endpoint that is plainly reachable.
 */
import { CopilotRuntime, createCopilotEndpoint } from "@copilotkit/runtime/v2";
import { HttpAgent } from "@ag-ui/client";

const AGUI_URL = process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui";

// Built per request, because the customer identity travels on a header and
// the agent must carry it downstream. A module-level agent would freeze
// whichever identity happened to load the module first.
const buildEndpoint = (user: string) => {
  const protec = new HttpAgent({
    url: AGUI_URL,
    headers: { "x-mcb-user": user },
  });
  return createCopilotEndpoint({
    runtime: new CopilotRuntime({ agents: { default: protec, protec } }),
    basePath: "/api/copilotkit",
  });
};

const handler = (req: Request) =>
  buildEndpoint(req.headers.get("x-mcb-user") ?? "anonymous").fetch(req);

export const GET = handler;
export const POST = handler;
export const OPTIONS = handler;

// Streaming, so this must not be pre-rendered or cached.
export const dynamic = "force-dynamic";
