"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { DialogueItem } from "@/types";
import { SPEAKER_COLORS } from "@/types";

interface TranscriptDisplayProps {
  items: DialogueItem[];
}

export function TranscriptDisplay({ items }: TranscriptDisplayProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Transcript</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] rounded-md">
          <div className="space-y-3 pr-4">
            {items.map((item, index) => (
              <div
                key={index}
                className={`rounded-lg border-l-4 p-3 ${
                  SPEAKER_COLORS[item.speaker as keyof typeof SPEAKER_COLORS] ||
                  ""
                }`}
              >
                <p className="font-semibold text-sm">{item.speaker}:</p>
                <p className="text-sm mt-1">{item.text}</p>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
