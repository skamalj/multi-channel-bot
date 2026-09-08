import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Protec console",
  description: "Glass-box console for the Protec multi-channel bot",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
