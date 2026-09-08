import type { ReactNode } from "react";
import "@copilotkit/react-core/v2/styles.css";
import "./globals.css";

export const metadata = {
  title: "Protec console",
  description: "Glass-box console for the Protec multi-channel bot",
};

// No provider here. It lives in the page because `threadId` is prop-controlled
// and the identity is chosen in the UI - a provider in a server layout cannot
// hold that state, and two providers would fight over the same agent.
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
