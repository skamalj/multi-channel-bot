import type { ReactNode } from "react";
import { CopilotKitProvider } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import "./globals.css";

export const metadata = {
  title: "Protec console",
  description: "Glass-box console for the Protec multi-channel bot",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* The agent name matches the key in the runtime's `agents` map. */}
        <CopilotKitProvider runtimeUrl="/api/copilotkit" agent="protec">
          {children}
        </CopilotKitProvider>
      </body>
    </html>
  );
}
