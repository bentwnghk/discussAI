"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { TranscriptDisplay } from "@/components/discuss/transcript-display";
import type { DialogueItem } from "@/types";
import { Headphones, AlertCircle } from "lucide-react";

interface SessionData {
  title: string;
  transcript: DialogueItem[];
  audioUrl: string;
  audioExpiresAt: string;
  audioExpired: boolean;
}

export default function ListenPage() {
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<SessionData | null>(null);
  const [submittedCode, setSubmittedCode] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return;

    setLoading(true);
    setError(null);
    setSession(null);
    setSubmittedCode("");

    const upperCode = code.trim().toUpperCase();

    try {
      const res = await fetch(`/api/public/session?code=${encodeURIComponent(upperCode)}`);
      if (!res.ok) {
        const data = await res.json();
        setError(data.error || "Invalid access code.");
        return;
      }
      setSession(await res.json());
      setSubmittedCode(upperCode);
    } catch {
      setError("Failed to fetch session.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-3xl">
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold flex items-center justify-center gap-2">
            <Headphones className="size-8" />
            Listen to Discussion
          </h1>
          <p className="text-muted-foreground">
            Enter the access code from your Word document to listen to the discussion audio.
          </p>
        </div>

        <Card>
          <CardContent className="pt-6">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <Input
                value={code}
                onChange={(e) => setCode(e.target.value.toUpperCase())}
                placeholder="Enter access code (e.g. ABC123)"
                className="text-center text-lg tracking-widest font-mono"
                maxLength={6}
              />
              <Button type="submit" disabled={loading || !code.trim()}>
                {loading ? "Loading..." : "Listen"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6 flex items-center gap-2 text-destructive">
              <AlertCircle className="size-5 shrink-0" />
              <p>{error}</p>
            </CardContent>
          </Card>
        )}

        {session && (
          <>
            <Card>
              <CardHeader>
                <CardTitle>{session.title}</CardTitle>
              </CardHeader>
              <CardContent>
                {session.audioExpired ? (
                  <p className="text-muted-foreground">Audio has expired.</p>
                ) : (
                  <audio controls className="w-full" src={`/api/public/audio?code=${encodeURIComponent(submittedCode)}`}>
                    Your browser does not support the audio element.
                  </audio>
                )}
              </CardContent>
            </Card>

            <TranscriptDisplay items={session.transcript} />
          </>
        )}
      </div>
    </div>
  );
}
