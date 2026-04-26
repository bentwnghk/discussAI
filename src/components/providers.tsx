"use client";

import { SessionProvider } from "next-auth/react";
import { Toaster } from "@/components/ui/sonner";
import { ApiKeyProvider } from "@/hooks/use-api-key";
import { CreditsProvider } from "@/hooks/use-credits";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <ApiKeyProvider>
        <CreditsProvider>{children}</CreditsProvider>
      </ApiKeyProvider>
      <Toaster position="top-right" richColors />
    </SessionProvider>
  );
}
