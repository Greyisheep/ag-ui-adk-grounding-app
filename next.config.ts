import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  env: {
    NEXT_PUBLIC_COPILOT_LICENSE_KEY: process.env.NEXT_PUBLIC_COPILOT_LICENSE_KEY,
  },
};

export default nextConfig;
