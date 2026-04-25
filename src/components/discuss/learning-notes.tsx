"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { LearningNotes as LearningNotesType } from "@/types";

interface LearningNotesProps {
  notes: LearningNotesType;
}

export function LearningNotes({ notes }: LearningNotesProps) {
  return (
    <div className="space-y-4">
      <Card className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10">
        <CardHeader>
          <CardTitle className="text-center text-2xl">
            📚 Study Notes 學習筆記
          </CardTitle>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-indigo-600">💡 Ideas 討論要點</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className="prose prose-sm max-w-none dark:prose-invert"
            dangerouslySetInnerHTML={{ __html: notes.ideas }}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-purple-600">📖 Language 語言學習</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className="prose prose-sm max-w-none dark:prose-invert [&_table]:w-full [&_table]:border-collapse [&_th]:bg-indigo-600 [&_th]:text-white [&_th]:px-3 [&_th]:py-2 [&_td]:border [&_td]:border-gray-200 [&_td]:px-3 [&_td]:py-2 [&_tr:nth-child(even)]:bg-gray-50"
            dangerouslySetInnerHTML={{ __html: notes.language }}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-pink-600">
            💬 Communication Strategies 溝通策略
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className="prose prose-sm max-w-none dark:prose-invert"
            dangerouslySetInnerHTML={{ __html: notes.communication_strategies }}
          />
        </CardContent>
      </Card>
    </div>
  );
}
