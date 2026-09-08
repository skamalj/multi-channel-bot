"use client";

/**
 * The console: assistant-ui chat on the left, the glass box on the right.
 *
 * WHY THE EXTERNAL-STORE RUNTIME. This build's whole point is the glass box,
 * and an abstraction that owns the transport hides the event stream from you.
 * With `useExternalStoreRuntime` the browser makes the request and reads the
 * SSE itself, so every AG-UI event is in hand: the answer goes to the chat,
 * and the CUSTOM trace events go to the panel, from ONE parse of ONE stream.
 *
 * That is not a stylistic preference. The previous runtime did not forward
 * CUSTOM events to subscribers at all, so the trace had to be polled from a
 * side-channel after the turn - the events were real but the liveness was
 * theatre. Here they arrive as they arrive.
 *
 * THE THREAD IS THE PERSON. `threadId` is the identity chosen below, which
 * becomes the resolver thread (ME-1) and, with the line of business appended,
 * the bot thread (ME-2). Nothing generates one on our behalf, so nothing can
 * silently give each turn a new conversation.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";

type TraceEvent = {
  kind: string;
  label: string;
  detail?: Record<string, unknown>;
};

type Msg = { id: string; role: "user" | "assistant"; text: string };
type Mode = { mode: "remote" | "local"; runtime_arn: string | null };

// The seeded customers and producer from the core store. A phone number is
// what a WhatsApp message arrives with, so it is what the console uses.
const IDENTITIES = [
  { id: "919820000009", label: "Priya Sharma - health + motor" },
  { id: "919820000002", label: "Rohit Verma - senior health" },
  { id: "919820000003", label: "Anita Desai - no policies" },
  { id: "919820000001", label: "Rakesh Nair - producer" },
];

const AGUI = "/api/agui";

export default function Page() {
  const [userId, setUserId] = useState(IDENTITIES[0].id);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem("mcb.userId");
    if (saved) setUserId(saved);
    setReady(true);
  }, []);

  useEffect(() => {
    if (ready) window.localStorage.setItem("mcb.userId", userId);
  }, [userId, ready]);

  if (!ready) return null;
  // Remounted per identity: one person's transcript must never appear to be
  // another's.
  return <Console key={userId} userId={userId} onUserChange={setUserId} />;
}

function Console({
  userId,
  onUserChange,
}: {
  userId: string;
  onUserChange: (id: string) => void;
}) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<Mode | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/mode")
      .then((r) => r.json())
      .then(setMode)
      .catch(() => setMode(null));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [events, messages]);

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const text = message.content
        .map((p) => (p.type === "text" ? p.text : ""))
        .join("")
        .trim();
      if (!text) return;

      const userMsg: Msg = { id: crypto.randomUUID(), role: "user", text };
      const assistantId = crypto.randomUUID();
      setMessages((m) => [...m, userMsg]);
      // A new turn starts a new trace. Keeping the previous turn's events
      // would make it look as though this answer consulted documents it
      // never touched.
      setEvents([]);
      setRunning(true);

      try {
        const res = await fetch(AGUI, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            threadId: userId,
            runId: userMsg.id,
            messages: [{ id: userMsg.id, role: "user", content: text }],
            state: {},
            tools: [],
            context: [],
            forwardedProps: { channel: "webchat" },
          }),
        });
        if (!res.body) throw new Error("no response body");

        // Parse the SSE ourselves. Frames are separated by a blank line and a
        // frame can be split across reads, so the tail is carried over rather
        // than assuming one chunk is one event.
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let answered = false;

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let cut: number;
          while ((cut = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, cut);
            buffer = buffer.slice(cut + 2);
            for (const line of frame.split("\n")) {
              if (!line.startsWith("data:")) continue;
              let ev: any;
              try {
                ev = JSON.parse(line.slice(5).trim());
              } catch {
                continue;
              }
              handle(ev);
            }
          }
        }

        function handle(ev: any) {
          switch (ev.type) {
            case "CUSTOM":
              if (ev.name === "trace" && ev.value) {
                setEvents((prev) => [...prev, ev.value as TraceEvent]);
              }
              break;
            case "TOOL_CALL_START":
              setEvents((prev) => [
                ...prev,
                { kind: "tool", label: String(ev.toolCallName ?? "") },
              ]);
              break;
            case "TEXT_MESSAGE_CONTENT":
              answered = true;
              setMessages((m) => {
                const has = m.some((x) => x.id === assistantId);
                if (!has) {
                  return [
                    ...m,
                    { id: assistantId, role: "assistant", text: ev.delta ?? "" },
                  ];
                }
                return m.map((x) =>
                  x.id === assistantId ? { ...x, text: x.text + (ev.delta ?? "") } : x,
                );
              });
              break;
            case "RUN_ERROR":
              setEvents((prev) => [
                ...prev,
                {
                  kind: "error",
                  label: String(ev.code ?? "RUN_ERROR"),
                  detail: { message: ev.message },
                },
              ]);
              break;
          }
        }

        if (!answered) {
          setMessages((m) => [
            ...m,
            {
              id: assistantId,
              role: "assistant",
              text: "(the agent produced no reply for this turn)",
            },
          ]);
        }
      } catch (err) {
        setMessages((m) => [
          ...m,
          {
            id: assistantId,
            role: "assistant",
            text: `Could not reach the agent: ${String(err)}`,
          },
        ]);
      } finally {
        setRunning(false);
      }
    },
    [userId],
  );

  const runtime = useExternalStoreRuntime<Msg>({
    messages,
    isRunning: running,
    setMessages,
    onNew,
    convertMessage: (m): ThreadMessageLike => ({
      role: m.role,
      content: [{ type: "text", text: m.text }],
      id: m.id,
    }),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="shell">
        <div className="chat">
          <ThreadPrimitive.Root className="thread">
            <ThreadPrimitive.Viewport className="viewport">
              <ThreadPrimitive.Empty>
                <div className="empty">
                  <h1>Protec</h1>
                  <p>
                    Ask about cover, a policy, a quote or a claim. Everything
                    the agent does appears on the right.
                  </p>
                </div>
              </ThreadPrimitive.Empty>

              <ThreadPrimitive.Messages
                components={{
                  UserMessage: () => (
                    <div className="msg user">
                      <MessagePrimitive.Parts />
                    </div>
                  ),
                  AssistantMessage: () => (
                    <div className="msg assistant">
                      <MessagePrimitive.Parts />
                    </div>
                  ),
                }}
              />

              {running && <div className="thinking">working…</div>}
              <div ref={bottom} />
            </ThreadPrimitive.Viewport>

            <ComposerPrimitive.Root className="composer">
              <ComposerPrimitive.Input
                className="composer-input"
                placeholder="Ask about your cover…"
                autoFocus
              />
              <ComposerPrimitive.Send className="composer-send">
                Send
              </ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
          </ThreadPrimitive.Root>
        </div>

        <aside className="glass">
          <h2>Glass box</h2>
          <p className="sub">Every AG-UI event this turn produced, in order.</p>

          <label className="who">
            <span>speaking as</span>
            <select
              value={userId}
              onChange={(e) => onUserChange(e.target.value)}
            >
              {IDENTITIES.map((i) => (
                <option key={i.id} value={i.id}>
                  {i.label}
                </option>
              ))}
            </select>
          </label>
          <p className="sub">
            thread <code>{userId}</code> - the resolver thread, and with the
            line of business appended, the bot thread.
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
        </aside>
      </div>
    </AssistantRuntimeProvider>
  );
}
