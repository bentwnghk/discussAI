"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApiKey } from "@/hooks/use-api-key";

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const { apiKey, setApiKey } = useApiKey();
  const [draft, setDraft] = useState(apiKey);

  useEffect(() => {
    if (open) setDraft(apiKey);
  }, [open, apiKey]);

  const handleSave = () => {
    setApiKey(draft);
    onOpenChange(false);
  };

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
            placeholder="API Key (sk-...)"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
          />
        </div>
        <DialogFooter>
          <Button onClick={handleSave}>Save</Button>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
