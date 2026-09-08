"use client";

/**
 * The console: CopilotKit's chat on the left, the glass box on the right.
 *
 * THE THREAD IS THE PERSON, and that is not decoration - the whole memory
 * design rests on it. `threadId` becomes the resolver thread (ME-1) and,
 * with the line of business appended, the bot thread (ME-2). Left to itself
 * CopilotKit mints a fresh UUID per run, so consecutive turns landed on
 * different threads and the agent could not remember the previous sentence.
 *
 * So the provider is prop-controlled here: the customer identity is chosen in
 * the UI, persisted, and passed as `threadId`. Switch identity and you are a
 * different person to the agent, with a different history - which is the
 * behaviour worth demonstrating.
 */
import { useEffect, useRef, useState } from "react";
import {
  CopilotChat,
  CopilotKitProvider,
  useAgent,
} from "@copilotkit/react-core/v2";

type TraceEvent = {
  kind: string;
  label: string;
  detail?: Record<string, unknown>;
};

type Mode = { mode: "remote" | "local"; runtime_arn: string | null };

// The seeded customers and producer from the core store. A phone number is
// what a WhatsApp message would arrive with, so it is what the console uses.
const IDENTITIES = [
  { id: "919820000009", label: "Priya Sharma - health + motor" },
  { id: "919820000002", label: "Rohit Verma - senior health" },
  { id: "919820000003", label: "Anita Desai - no policies" },
  { id: "919820000001", label: "Rakesh Nair - producer" },
];

export default function Page() {
  const [userId, setUserId] = useState<string>(IDENTITIES[0].id);
  const [ready, setReady] = useState(false);

  // Restored so a reload does not silently become a different customer.
  useEffect(() => {
    const saved = window.localStorage.getItem("mcb.userId");
    if (saved) setUserId(saved);
    setReady(true);
  }, []);

  useEffect(() => {
    if (ready) window.localStorage.setItem("mcb.userId", userId);
  }, [userId, ready]);

  if (!ready) return null;

  return (
    <CopilotKitProvider
      runtimeUrl="/api/copilotkit"
      agent="protec"
      threadId={userId}
      // The header is what actually reaches the agent. CopilotKit's threadId
      // governs its own transcript; the AG-UI payload it sends onward
      // carries an internal UUID, so the identity is stated explicitly.
      headers={{ "x-mcb-user": userId }}
      // Remounted on change so the transcript belongs to the identity it was
      // produced under, rather than one person's history appearing to be
      // another's.
      key={userId}
    >
      <Console userId={userId} onUserChange={setUserId} />
    </CopilotKitProvider>
  );
}

function Console({
  userId,
  onUserChange,
}: {
  userId: string;
  onUserChange: (id: string) => void;
}) {
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

  // CopilotKit does not forward AG-UI CUSTOM events to a subscriber, so the
  // trace is read from the bridge, which kept what it produced. The events
  // and their order are the real ones; only the liveness is not - they land
  // when the turn ends rather than as it runs.
  //
  // Keyed by identity now that the thread is stable, so switching customer
  // shows that customer's last turn rather than whoever happened to go last.
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const r = await fetch(`/api/trace/${encodeURIComponent(userId)}`, {
          cache: "no-store",
        });
        const body = await r.json();
        if (!stop && Array.isArray(body.events)) {
          setEvents(body.events as TraceEvent[]);
        }
      } catch {
        /* the bridge is not up; the chat will say so on its own */
      }
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [userId, agent]);

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
        <p className="sub">Every AG-UI event this turn produced, in order.</p>

        <label className="who">
          <span>speaking as</span>
          <select value={userId} onChange={(e) => onUserChange(e.target.value)}>
            {IDENTITIES.map((i) => (
              <option key={i.id} value={i.id}>
                {i.label}
              </option>
            ))}
          </select>
        </label>
        <p className="sub">
          thread <code>{userId}</code> - the resolver thread, and with the line
          of business appended, the bot thread.
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
