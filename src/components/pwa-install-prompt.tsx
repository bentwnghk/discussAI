"use client";

import { useState, useEffect } from "react";
import { usePWAInstall } from "react-use-pwa-install";
import { Download, X, Share, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

const wasDismissed =
  typeof window !== "undefined" &&
  sessionStorage.getItem("pwa-prompt-dismissed") === "true";

function isIOS() {
  if (typeof window === "undefined") return false;
  return /iPad|iPhone|iPod/.test(navigator.userAgent);
}

function isStandalone() {
  if (typeof window === "undefined") return false;
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    (window.navigator as unknown as { standalone?: boolean }).standalone ===
      true
  );
}

export function PWAInstallPrompt() {
  const install = usePWAInstall();
  const [dismissed, setDismissed] = useState(wasDismissed);
  const [showIOS, setShowIOS] = useState(false);

  useEffect(() => {
    if (isIOS() && !isStandalone()) {
      const timer = setTimeout(() => setShowIOS(true), 3000);
      return () => clearTimeout(timer);
    }
  }, []);

  if (dismissed || isStandalone()) return null;

  if (showIOS && !install) {
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
          <div className="flex-1 space-y-3">
            <div>
              <p className="text-sm font-semibold">Install Mr.🆖 DiscussAI</p>
              <p className="text-xs text-muted-foreground">
                Add to your home screen for a native app experience.
              </p>
            </div>
            <ol className="space-y-1.5 text-xs text-muted-foreground">
              <li className="flex items-center gap-2">
                <Share className="size-3.5 shrink-0" />
                Tap the Share button in Safari
              </li>
              <li className="flex items-center gap-2">
                <Plus className="size-3.5 shrink-0" />
                Scroll down and tap &quot;Add to Home Screen&quot;
              </li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  if (!install) return null;

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
