import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    // Set this to your Modal deployment URL when deployed
    // The app name is "flux2-klein-realtime" so the URL will be:
    // wss://YOUR_USERNAME--flux2-klein-realtime-flux2kleinserver-web.modal.run
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || "wss://developedbyed--flux2-klein-realtime-flux2kleinserver-web.modal.run",
  },
};

export default nextConfig;
