"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface AudioPlayerProps {
  src: string | null;
}

export function AudioPlayer({ src }: AudioPlayerProps) {
  if (!src) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Audio</CardTitle>
      </CardHeader>
      <CardContent>
        <audio controls className="w-full" src={src}>
          Your browser does not support the audio element.
        </audio>
      </CardContent>
    </Card>
  );
}
