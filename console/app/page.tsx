"use client";

/**
 * The console: CopilotKit's chat on the left, the glass box on the right.
 *
 * The glass box is the reason this project exists, and it is what a plain
 * chat component cannot give you on its own. `useAgent` exposes the agent's
 * event stream, so every AG-UI event the agent emitted - the resolver's
 * routing decision, each tool call, the citation guardrail's verdict, the
 * grounding score - is rendered beside the answer as it arrives.
 *
 * The trace events arrive as AG-UI `CUSTOM` events named "trace". That is
 * deliberate: they are not part of the conversation and must never be
 * rendered as though the assistant said them.
 */
import { useEffect, useRef, useState } from "react";
import { CopilotChat, useAgent } from "@copilotkit/react-core/v2";

type TraceEvent = {
  kind: string;
  label: string;
  detail?: Record<string, unknown>;
};

type Mode = { mode: "remote" | "local"; runtime_arn: string | null };

export default function Page() {
  const { agent } = useAgent({ agentId: "protec" });
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [mode, setMode] = useState<Mode | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  // Which agent is actually answering. A console that quietly ran the agent
  // in-process would look identical to one driving the deployment and would
  // prove nothing, so it says which.
  useEffect(() => {
    fetch("/api/mode")
      .then((r) => r.json())
      .then(setMode)
      .catch(() => setMode(null));
  }, []);

  useEffect(() => {
    if (!agent) return;
    const record = (e: TraceEvent) => setEvents((prev) => [...prev, e]);
    const subscription: any = agent.subscribe({
      // Which callbacks a CopilotKit agent actually invokes is not obvious
      // from the types, so both the generic and the specific ones are wired
      // and whichever fires wins. `onCustomEvent` is the one that carries
      // our trace; `onEvent` is the catch-all.
      onCustomEvent: (p: any) => {
        const ev = p?.event ?? p;
        if (ev?.name === "trace" && ev?.value) record(ev.value as TraceEvent);
      },
      onToolCallStartEvent: (p: any) => {
        const ev = p?.event ?? p;
        record({ kind: "tool", label: String(ev?.toolCallName ?? "") });
      },
      onRunStartedEvent: () => setEvents([]),
      onEvent: ({ event }: { event: Record<string, unknown> }) => {
        const type = event?.type as string | undefined;

        // Only the cases the specific callbacks above do not cover.
        if (type === "RUN_STARTED") {
          // A new turn starts a new trace. Keeping the previous turn's
          // events would make it look as though this answer consulted
          // documents it never touched.
          setEvents([]);
        }
        if (type === "RUN_ERROR") {
          setEvents((prev) => [
            ...prev,
            {
              kind: "error",
              label: String((event as any).code ?? "RUN_ERROR"),
              detail: { message: (event as any).message },
            },
          ]);
        }
      },
    });
    // `subscribe` returns a subscription OBJECT, not a function. Calling
    // the return value directly throws "unsubscribe is not a function" and
    // leaves the listener attached, so every turn stacks another one.
    return () => {
      if (typeof subscription === "function") subscription();
      else subscription?.unsubscribe?.();
    };
  }, [agent]);

  // CopilotKit does not forward AG-UI CUSTOM events to a subscriber, so the
  // trace is read from the bridge, which kept what it produced. The events
  // and their order are the real ones; only the liveness is not - they land
  // when the turn ends rather than as it runs.
  useEffect(() => {
    if (!agent) return;
    // CopilotKit mints a fresh threadId per run, so there is no stable key
    // to ask for. "__last" is what this console actually wants.
    const thread = "__last";
    const tick = async () => {
      try {
        const r = await fetch(`/api/trace/${encodeURIComponent(thread)}`, {
          cache: "no-store",
        });
        const body = await r.json();
        if (Array.isArray(body.events) && body.events.length) {
          setEvents(body.events as TraceEvent[]);
        }
      } catch {
        /* the bridge is not up; the chat will say so on its own */
      }
    };
    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  }, [agent]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  return (
    <div className="shell">
      <div className="chat">
        <CopilotChat />
      </div>

      <aside className="glass">
        <h2>Glass box</h2>
        <p className="sub">
          Every AG-UI event this turn produced, in order.
        </p>

        {mode && (
          <span className={`mode ${mode.mode}`}>
            {mode.mode === "remote"
              ? "answering from the deployed AgentCore Runtime"
              : "answering in-process (no runtime ARN set)"}
          </span>
        )}

        {events.length === 0 && (
          <p className="sub">Ask something to see the agent work.</p>
        )}

        {events.map((e, i) => (
          <div key={i} className={`ev ${e.kind}`}>
            <span className="k">{e.kind}</span>{" "}
            <span className="l">{e.label}</span>
            {e.detail && Object.keys(e.detail).length > 0 && (
              <div className="d">{JSON.stringify(e.detail, null, 1)}</div>
            )}
          </div>
        ))}
        <div ref={bottom} />
      </aside>
    </div>
  );
}
