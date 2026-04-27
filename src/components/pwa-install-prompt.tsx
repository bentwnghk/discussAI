"use client";

import { useState, useEffect, useCallback } from "react";
import { Download, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { usePWAInstall } from "@/hooks/use-pwa-install";

const wasDismissed =
  typeof window !== "undefined" &&
  sessionStorage.getItem("pwa-prompt-dismissed") === "true";

export function PWAInstallPrompt() {
  const { isInstallable, isInstalled, install, registerSW } = usePWAInstall();
  const [dismissed, setDismissed] = useState(wasDismissed);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    registerSW().then(() => setReady(true));
  }, [registerSW]);

  if (!ready || isInstalled || dismissed || !isInstallable) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 z-[60] mx-auto max-w-sm rounded-xl border bg-background p-4 shadow-lg dark:shadow-black/40 md:left-auto md:right-4">
      <button
        onClick={() => {
          setDismissed(true);
          sessionStorage.setItem("pwa-prompt-dismissed", "true");
        }}
        className="absolute right-2 top-2 rounded-md p-1 text-muted-foreground hover:text-foreground"
        aria-label="Dismiss"
      >
        <X className="size-4" />
      </button>
      <div className="flex items-start gap-3">
        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          <Download className="size-5 text-primary" />
        </div>
        <div className="flex-1 space-y-2">
          <div>
            <p className="text-sm font-semibold">Install Mr.🆖 DiscussAI</p>
            <p className="text-xs text-muted-foreground">
              Add to your home screen for a native app experience.
            </p>
          </div>
          <Button size="sm" onClick={install} className="w-full">
            <Download className="size-4" />
            Install App
          </Button>
        </div>
      </div>
    </div>
  );
}
