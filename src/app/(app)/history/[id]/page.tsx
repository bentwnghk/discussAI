"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { TranscriptDisplay } from "@/components/discuss/transcript-display";
import { LearningNotes } from "@/components/discuss/learning-notes";
import { AudioPlayer } from "@/components/discuss/audio-player";
import type { DialogueItem, LearningNotes as LearningNotesType } from "@/types";
import { FileText, ArrowLeft } from "lucide-react";

interface SessionData {
  id: string;
  title: string;
  dialogueMode: string;
  inputMethod: string;
  inputText: string | null;
  transcript: DialogueItem[];
  learningNotes: LearningNotesType;
  audioUrl: string | null;
  audioExpiresAt: string | null;
  accessCode: string | null;
  charactersCount: number;
  ttsCostHKD: number;
  usedOwnApiKey: boolean;
  generationCost: number;
  createdAt: string;
}

export default function SessionDetailPage() {
  const params = useParams();
  const [session, setSession] = useState<SessionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [audioDownloadName, setAudioDownloadName] = useState<string | undefined>();
  const [expiryDays, setExpiryDays] = useState<number | null>(null);
  const [docxDownloadName, setDocxDownloadName] = useState<string | undefined>();

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`/api/history/${params.id}`);
        if (!res.ok) throw new Error("Not found");
        const data = await res.json();
        setSession(data);

        if (data.audioUrl) {
          setAudioSrc(data.audioUrl);
        }

        const timestamp = new Date(data.createdAt).toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" }).replace(/[/:, ]/g, "-");
        setAudioDownloadName(`Mr.NG-DiscussAI-audio-${timestamp}.mp3`);
        setDocxDownloadName(`Mr.NG-DiscussAI-notes-${timestamp}.docx`);

        if (data.audioExpiresAt) {
          setExpiryDays((new Date(data.audioExpiresAt).getTime() - Date.now()) / (24 * 60 * 60 * 1000));
        }
      } catch {
        // session not found
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [params.id]);

  const handleExportDocx = useCallback(async () => {
    if (!session) return;
    try {
      const res = await fetch("/api/export/docx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: session.transcript,
          learningNotes: session.learningNotes,
          title: session.title,
          extractedText: session.inputText,
          accessCode: session.accessCode,
        }),
      });
      if (!res.ok) throw new Error("Export failed.");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = docxDownloadName || "Mr.NG-DiscussAI-notes.docx";
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // silently fail
    }
  }, [session, docxDownloadName]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        <p className="text-center text-muted-foreground">Loading...</p>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">Session not found.</p>
            <Link href="/history">
              <Button className="mt-4">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to History
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{session.title}</h1>
            <p className="text-sm text-muted-foreground">
              {new Date(session.createdAt).toLocaleString("en-HK", {
                timeZone: "Asia/Hong_Kong",
              })}
              {" | "}
              {session.usedOwnApiKey
                ? `HK$${session.ttsCostHKD.toFixed(2)}`
                : `${session.generationCost} credits`}
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleExportDocx}>
              <FileText className="mr-2 h-4 w-4" />
              Export to Word
            </Button>
            <Link href="/history">
              <Button variant="outline">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back
              </Button>
            </Link>
          </div>
        </div>

        <Separator />

        {session.inputText && (
          <Card>
            <CardHeader>
              <CardTitle>📌 Task 任務</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-64 overflow-y-auto rounded-md bg-muted p-4">
                <pre className="whitespace-pre-wrap text-sm leading-relaxed font-sans">
                  {session.inputText}
                </pre>
              </div>
            </CardContent>
          </Card>
        )}

        <AudioPlayer
          src={audioSrc}
          expiryDays={expiryDays}
          accessCode={session.accessCode}
          downloadFileName={audioDownloadName}
        />

        <TranscriptDisplay items={session.transcript} />

        <LearningNotes notes={session.learningNotes} />
      </div>
    </div>
  );
}
