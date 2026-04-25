"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useApiKey } from "@/hooks/use-api-key";

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { apiKey, setApiKey } = useApiKey();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Configure your application settings.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Get your API key{" "}
            <a
              href="https://api.mr5ai.com"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              here
            </a>
          </p>
          <Input
            type="password"
            placeholder="API Key (optional if set server-side)"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
