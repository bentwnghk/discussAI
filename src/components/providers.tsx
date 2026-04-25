"use client";

import { SessionProvider } from "next-auth/react";
import { Toaster } from "@/components/ui/sonner";
import { ApiKeyProvider } from "@/hooks/use-api-key";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <ApiKeyProvider>{children}</ApiKeyProvider>
      <Toaster position="top-right" richColors />
    </SessionProvider>
  );
}
