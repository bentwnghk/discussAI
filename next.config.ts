import type { NextConfig } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants.js";
import withSerwistInit from "@serwist/next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["pdf-parse"],
  transpilePackages: ["pdfjs-dist"],
  turbopack: {},
};

export default (phase: string) => {
  if (phase === PHASE_PRODUCTION_BUILD) {
    const withSerwist = withSerwistInit({
      swSrc: "src/app/sw.ts",
      swDest: "public/sw.js",
      register: false,
    });
    return withSerwist(nextConfig);
  }

  return nextConfig;
};
