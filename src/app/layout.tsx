import type { Metadata } from "next";

import { CopilotKit } from "@copilotkit/react-core";
import "./globals.css";
import "@copilotkit/react-ui/styles.css";

export const metadata: Metadata = {
  title: "AI Agent with Google ADK & CopilotKit",
  description: "Full-stack AI agent with Google Search and Maps grounding, built with ADK, CopilotKit, and AG-UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={"antialiased"}>
        <CopilotKit 
          runtimeUrl="/api/copilotkit" 
          agent="my_agent" 
          publicLicenseKey={process.env.NEXT_PUBLIC_COPILOT_LICENSE_KEY}
        >
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
