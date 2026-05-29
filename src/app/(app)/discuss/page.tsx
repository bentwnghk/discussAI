"use client";

import { useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { FileUpload } from "@/components/discuss/file-upload";
import { TranscriptDisplay } from "@/components/discuss/transcript-display";
import { LearningNotes } from "@/components/discuss/learning-notes";
import { AudioPlayer } from "@/components/discuss/audio-player";
import type {
  InputMethod,
  DialogueMode,
  DialogueItem,
  LearningNotes as LearningNotesType,
  GenerateResponse,
  Speaker,
} from "@/types";
import { Sparkles, FileText, History, Users } from "lucide-react";
import { getVoiceForSpeaker } from "@/lib/tts/generate";
import { processPdf } from "@/lib/pdf-client";
import { useCredits } from "@/hooks/use-credits";

type SseProgressEvent = {
  type: "progress";
  stage: string;
  message: string;
};

type SseCompleteEvent = {
  type: "complete";
  data: GenerateResponse;
};

type SseErrorEvent = {
  type: "error";
  message: string;
};

type SseEvent = SseProgressEvent | SseCompleteEvent | SseErrorEvent;

const STAGE_PROGRESS: Record<string, number> = {
  processing_files: 22,
  processing_file: 25,
  generating: 30,
  finalizing: 37,
};

export default function DiscussPage() {
  const router = useRouter();
  const { refreshBalance } = useCredits();
  const [inputMethod, setInputMethod] = useState<InputMethod>("Upload Files");
  const [dialogueMode, setDialogueMode] = useState<DialogueMode>("Normal");
  const [files, setFiles] = useState<File[]>([]);
  const [topicText, setTopicText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [generatingStage, setGeneratingStage] = useState<string | null>(null);

  const [dialogueItems, setDialogueItems] = useState<DialogueItem[]>([]);
  const [learningNotes, setLearningNotes] = useState<LearningNotesType | null>(
    null
  );
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [sessionTitle, setSessionTitle] = useState<string>("");
  const [charactersCount, setCharactersCount] = useState(0);
  const [ttsCostHKD, setTtsCostHKD] = useState(0);
  const [extractedText, setExtractedText] = useState<string | null>(null);
  const [usedOwnApiKey, setUsedOwnApiKey] = useState(false);
  const [creditsConsumed, setCreditsConsumed] = useState(0);
  const [accessCode, setAccessCode] = useState<string | null>(null);
  const [expiryDays, setExpiryDays] = useState<number | null>(null);
  const [generatedTimestamp, setGeneratedTimestamp] = useState<string | null>(null);

  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startElapsedTimer = useCallback(() => {
    setElapsedSeconds(0);
    elapsedTimerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);
  }, []);

  const stopElapsedTimer = useCallback(() => {
    if (elapsedTimerRef.current) {
      clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = null;
    }
    setElapsedSeconds(0);
  }, []);

  const handleGenerate = useCallback(async () => {
    if (
      inputMethod === "Upload Files" &&
      files.length === 0
    ) {
      toast.error("Please upload at least one file.");
      return;
    }
    if (inputMethod === "Enter Topic" && !topicText.trim()) {
      toast.error("Please enter a topic.");
      return;
    }

    setIsGenerating(true);
    setProgress(5);
    setProgressLabel("Preparing files...");
    setGeneratingStage(null);

    let generationId: string | null = null;

    try {
      const formData = new FormData();
      formData.append("inputMethod", inputMethod);
      formData.append("dialogueMode", dialogueMode);
      if (inputMethod === "Enter Topic") {
        formData.append("text", topicText);
      }
      if (inputMethod === "Upload Files") {
        const pdfFiles = files.filter(
          (f) =>
            f.type === "application/pdf" ||
            f.name.toLowerCase().endsWith(".pdf")
        );
        const nonPdfFiles = files.filter((f) => !pdfFiles.includes(f));

        nonPdfFiles.forEach((f) => formData.append("files", f));

        if (pdfFiles.length > 0) {
          setProgressLabel("Processing PDF pages...");
          setProgress(8);
          for (const pdfFile of pdfFiles) {
            try {
              const result = await processPdf(pdfFile);
              formData.append("fileName", result.fileName);
              if (result.text) {
                formData.append("preExtractedText", result.text);
              } else {
                result.images.forEach((img) =>
                  formData.append("files", img)
                );
              }
            } catch {
              formData.append("files", pdfFile);
            }
          }
        }
      }

      setProgressLabel("Connecting to server...");
      setProgress(20);

      const res = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let errorMsg = "Generation failed.";
        try {
          const err = await res.json();
          if (res.status === 402) {
            throw new Error(
              `Insufficient credits. You need ${err.creditsNeeded} credits but have ${err.currentBalance}. Go to Credits page to purchase more.`
            );
          }
          errorMsg = err.error || errorMsg;
        } catch (jsonError) {
          if (
            jsonError instanceof Error &&
            jsonError.message.startsWith("Insufficient credits")
          ) {
            throw jsonError;
          }
          errorMsg =
            "Server returned an unexpected response. Your credits will be refunded if they were deducted.";
        }
        throw new Error(errorMsg);
      }

      startElapsedTimer();

      let data: GenerateResponse | null = null;
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const event: SseEvent = JSON.parse(line.slice(6));

          if (event.type === "progress") {
            const pct = STAGE_PROGRESS[event.stage] ?? 25;
            setProgress(pct);
            setProgressLabel(event.message);
            setGeneratingStage(event.stage);
          } else if (event.type === "complete") {
            data = event.data;
          } else if (event.type === "error") {
            stopElapsedTimer();
            throw new Error(event.message);
          }
        }
      }

      stopElapsedTimer();

      if (!data) {
        throw new Error("Server returned an incomplete response.");
      }

      generationId = data.generationId ?? null;
      setDialogueItems(data.dialogue);
      setLearningNotes(data.learningNotes);
      setSessionTitle(data.title);
      setCharactersCount(data.charactersCount);
      setTtsCostHKD(data.ttsCostHKD);
      setExtractedText(data.extractedText);
      setUsedOwnApiKey(data.usedOwnApiKey);
      setCreditsConsumed(data.creditsConsumed);

      if (!data.usedOwnApiKey) {
        refreshBalance();
      }

      setProgressLabel("Generating audio...");
      setProgress(40);

      let audioFailed = false;
      const totalLines = data.dialogue.length;
      const audioChunks: ArrayBuffer[] = [];
      let completed = 0;

      try {
        const batchSize = 5;
        for (let i = 0; i < totalLines; i += batchSize) {
          const batch = data.dialogue.slice(i, i + batchSize);
          const promises = batch.map(async (item) => {
            const voice = getVoiceForSpeaker(item.speaker as Speaker);
            const ttsRes = await fetch("/api/tts", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                text: item.text,
                voice,
              }),
            });
            if (!ttsRes.ok) throw new Error("TTS failed for a line.");
            return ttsRes.arrayBuffer();
          });

          const results = await Promise.all(promises);
          audioChunks.push(...results);
          completed += batch.length;
          const pct = 40 + Math.round((completed / totalLines) * 50);
          setProgress(pct);
          setProgressLabel(`Generating audio ${completed}/${totalLines}...`);
        }
      } catch {
        audioFailed = true;
      }

      if (audioFailed) {
        if (generationId) {
          await fetch("/api/credits/refund", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ generationId }),
          }).catch(() => {});
        }
        refreshBalance();
        throw new Error("Audio generation failed. Your credits have been refunded.");
      }

      const totalLength = audioChunks.reduce(
        (sum, chunk) => sum + chunk.byteLength,
        0
      );
      const combined = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of audioChunks) {
        combined.set(new Uint8Array(chunk), offset);
        offset += chunk.byteLength;
      }

      const blob = new Blob([combined], { type: "audio/mpeg" });
      const objectUrl = URL.createObjectURL(blob);
      setAudioUrl(objectUrl);

      setProgressLabel("Saving session...");
      setProgress(95);

      let savedAudioUrl: string | null = null;
      try {
        const uploadForm = new FormData();
        uploadForm.append("audio", blob, "discussion.mp3");
        const uploadRes = await fetch("/api/upload-audio", {
          method: "POST",
          body: uploadForm,
        });
        if (uploadRes.ok) {
          const { audioUrl: serverUrl } = await uploadRes.json();
          savedAudioUrl = serverUrl;
        }
      } catch {
        console.warn("Audio upload failed, session will be saved without audio.");
      }

      try {
        const saveRes = await fetch("/api/history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: data.title,
            sessionType: "discussion",
            dialogueMode,
            inputMethod,
            inputText: data.extractedText,
            transcript: data.dialogue,
            learningNotes: data.learningNotes,
            audioUrl: savedAudioUrl,
            charactersCount: data.charactersCount,
            ttsCostHKD: data.ttsCostHKD,
            usedOwnApiKey: data.usedOwnApiKey,
          }),
        });
        if (saveRes.ok) {
          const saved = await saveRes.json();
          if (saved.accessCode) setAccessCode(saved.accessCode);
          if (saved.audioExpiresAt) {
            setExpiryDays((new Date(saved.audioExpiresAt).getTime() - Date.now()) / (24 * 60 * 60 * 1000));
          }
          if (saved.createdAt) {
            setGeneratedTimestamp(new Date(saved.createdAt).toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" }).replace(/[/:, ]/g, "-"));
          }
        }
      } catch {
        if (generationId) {
          await fetch("/api/credits/refund", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ generationId }),
          }).catch(() => {});
        }
        refreshBalance();
        toast.warning("Failed to save session. Your credits have been refunded.");
        return;
      }

      setProgress(100);
      setProgressLabel("Done!");
      if (data.usedOwnApiKey) {
        toast.success(
          `Discussion generated! Total cost: HK$${data.ttsCostHKD.toFixed(2)}`
        );
      } else {
        toast.success(
          `Discussion generated! Credits consumed: ${data.creditsConsumed}`
        );
      }
    } catch (error: unknown) {
      stopElapsedTimer();
      const message =
        error instanceof Error ? error.message : "An error occurred.";
      toast.error(message);

      if (generationId) {
        await fetch("/api/credits/refund", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ generationId }),
        }).catch(() => {});
      } else {
        await fetch("/api/credits/refund", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refundLast: true }),
        }).catch(() => {});
      }
      refreshBalance();
    } finally {
      stopElapsedTimer();
      setIsGenerating(false);
    }
  }, [inputMethod, dialogueMode, files, topicText, refreshBalance, startElapsedTimer, stopElapsedTimer]);

  const handleExportDocx = useCallback(async () => {
    if (!dialogueItems.length || !learningNotes) return;
    try {
      const res = await fetch("/api/export/docx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: dialogueItems,
          learningNotes,
          title: sessionTitle,
          extractedText,
          accessCode,
        }),
      });
      if (!res.ok) throw new Error("Export failed.");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Mr.NG-DiscussAI-notes-${generatedTimestamp || "unknown"}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Document downloaded!");
    } catch {
      toast.error("Failed to export document.");
    }
  }, [dialogueItems, learningNotes, sessionTitle, extractedText, accessCode, generatedTimestamp]);

  const audioDownloadName = generatedTimestamp
    ? `Mr.NG-DiscussAI-audio-${generatedTimestamp}.mp3`
    : undefined;

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="size-6" /> Group Discussion Topic
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <Label className="mb-2 block">Input Method</Label>
              <RadioGroup
                value={inputMethod}
                onValueChange={(v) => setInputMethod(v as InputMethod)}
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Upload Files" id="upload" />
                  <Label htmlFor="upload">Upload Files</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Enter Topic" id="text" />
                  <Label htmlFor="text">Enter Topic</Label>
                </div>
              </RadioGroup>
            </div>

            {inputMethod === "Upload Files" ? (
              <FileUpload files={files} onFilesChange={setFiles} />
            ) : (
              <Textarea
                placeholder="Paste or type your discussion topic here..."
                value={topicText}
                onChange={(e) => setTopicText(e.target.value)}
                rows={10}
              />
            )}

            <div>
              <Label className="mb-2 block">Depth of Discussion</Label>
              <RadioGroup
                value={dialogueMode}
                onValueChange={(v) => setDialogueMode(v as DialogueMode)}
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Normal" id="normal" />
                  <Label htmlFor="normal">Normal</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Deeper" id="deeper" />
                  <Label htmlFor="deeper">
                    Deeper
                    <span className="text-xs text-muted-foreground ml-1">
                      (More ideas & elaboration)
                    </span>
                  </Label>
                </div>
              </RadioGroup>
            </div>

            {isGenerating && (
              <div className="space-y-2">
                <Progress value={progress} />
                <p className="text-sm text-muted-foreground text-center">
                  {progressLabel}
                  {elapsedSeconds > 0 && generatingStage === "generating" && (
                    <span className="text-muted-foreground/70 ml-1">
                      ({elapsedSeconds}s elapsed)
                    </span>
                  )}
                </p>
              </div>
            )}

            <Button
              onClick={handleGenerate}
              disabled={isGenerating}
              size="lg"
              className="w-full"
            >
              {isGenerating ? (
                "Generating..."
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate Discussion
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {dialogueItems.length > 0 && (
          <>
            <Separator />

            {extractedText && (
              <Card>
                <CardHeader>
                  <CardTitle>📌 Task 任務</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="max-h-64 overflow-y-auto rounded-md bg-muted p-4">
                    <pre className="whitespace-pre-wrap text-sm leading-relaxed font-sans">
                      {extractedText}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            )}

            <AudioPlayer
              src={audioUrl}
              accessCode={accessCode}
              expiryDays={expiryDays}
              downloadFileName={audioDownloadName}
            />

            <TranscriptDisplay items={dialogueItems} />

            {learningNotes && <LearningNotes notes={learningNotes} />}

            <div className="flex justify-between items-center">
              <p className="text-sm text-muted-foreground">
                {usedOwnApiKey
                  ? `Cost: HK$${ttsCostHKD.toFixed(2)} | ${charactersCount} characters`
                  : `Credits spent: ${creditsConsumed} | ${charactersCount} characters`}
              </p>
              <Button onClick={handleExportDocx} variant="outline">
                <FileText className="mr-2 h-4 w-4" />
                Export to Word
              </Button>
            </div>
          </>
        )}

        <Separator />

        <div className="text-center">
          <Button
            variant="ghost"
            onClick={() => router.push("/history")}
          >
            <History className="mr-2 h-4 w-4" />
            View History
          </Button>
        </div>
      </div>
    </div>
  );
}
