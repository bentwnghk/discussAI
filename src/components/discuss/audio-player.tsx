"use client";

import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Clock, Download, KeyRound } from "lucide-react";

interface AudioPlayerProps {
  src: string | null;
  expiryDays?: number | null;
  accessCode?: string | null;
  downloadFileName?: string;
}

export function AudioPlayer({
  src,
  expiryDays,
  accessCode,
  downloadFileName,
}: AudioPlayerProps) {
  const expiryLabel = useMemo(() => {
    if (expiryDays == null) return null;
    return `Expires in ${Math.max(0, Math.ceil(expiryDays))} days`;
  }, [expiryDays]);

  if (!src) return null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>🎧 Audio 錄音</CardTitle>
          <div className="flex items-center gap-2">
            {expiryLabel && (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                {expiryLabel}
              </span>
            )}
            {downloadFileName && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const a = document.createElement("a");
                  a.href = src;
                  a.download = downloadFileName;
                  a.click();
                }}
              >
                <Download className="mr-1 h-3 w-3" />
                Download
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <audio controls className="w-full" src={src}>
          Your browser does not support the audio element.
        </audio>
        {accessCode && (
          <div className="mt-3 flex items-center gap-2 rounded-md bg-muted px-3 py-2">
            <KeyRound className="size-4 text-muted-foreground shrink-0" />
            <span className="text-sm text-muted-foreground">Access Code:</span>
            <span className="font-mono font-bold tracking-widest text-sm">
              {accessCode}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
