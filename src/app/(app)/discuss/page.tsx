"use client";

import { useState, useCallback } from "react";
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
import { Sparkles, FileText, History } from "lucide-react";
import { getVoiceForSpeaker } from "@/lib/tts/generate";
import { processPdf } from "@/lib/pdf-client";
import { useCredits } from "@/hooks/use-credits";

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

      setProgressLabel("Generating transcript and study notes...");
      setProgress(20);

      const res = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        if (res.status === 402) {
          throw new Error(
            `Insufficient credits. You need ${err.creditsNeeded} credits but have ${err.currentBalance}. Go to Credits page to purchase more.`
          );
        }
        throw new Error(err.error || "Generation failed.");
      }

      const data: GenerateResponse = await res.json();
      const { generationId } = data;
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
        await fetch("/api/history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: data.title,
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
      const message =
        error instanceof Error ? error.message : "An error occurred.";
      toast.error(message);
      refreshBalance();
    } finally {
      setIsGenerating(false);
    }
  }, [inputMethod, dialogueMode, files, topicText, refreshBalance]);

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
        }),
      });
      if (!res.ok) throw new Error("Export failed.");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const timestamp = new Date().toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" }).replace(/[/:, ]/g, "-");
      a.download = `Mr.NG-DiscussAI-notes-${timestamp}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Document downloaded!");
    } catch {
      toast.error("Failed to export document.");
    }
  }, [dialogueItems, learningNotes, sessionTitle]);

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <span>✅</span> Discussion Topic
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <Label className="mb-2 block">📁 Input Method</Label>
              <RadioGroup
                value={inputMethod}
                onValueChange={(v) => setInputMethod(v as InputMethod)}
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Upload Files" id="upload" />
                  <Label htmlFor="upload">📎 Upload Files</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Enter Topic" id="text" />
                  <Label htmlFor="text">✏️ Enter Topic</Label>
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
              <Label className="mb-2 block">🧠 Depth of Discussion</Label>
              <RadioGroup
                value={dialogueMode}
                onValueChange={(v) => setDialogueMode(v as DialogueMode)}
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Normal" id="normal" />
                  <Label htmlFor="normal">🌿 Normal</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Deeper" id="deeper" />
                  <Label htmlFor="deeper">
                    🔍 Deeper
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

            <AudioPlayer src={audioUrl} />

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
            View Practice History
          </Button>
        </div>
      </div>
    </div>
  );
}
