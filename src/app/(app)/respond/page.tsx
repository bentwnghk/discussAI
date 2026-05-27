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
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { FileUpload } from "@/components/discuss/file-upload";
import { TranscriptDisplay } from "@/components/discuss/transcript-display";
import { LearningNotes } from "@/components/discuss/learning-notes";
import { AudioPlayer } from "@/components/discuss/audio-player";
import type {
  InputMethod,
  DialogueMode,
  DialogueItem,
  LearningNotes as LearningNotesType,
  RespondResponse,
  VoiceOption,
} from "@/types";
import { Sparkles, FileText, History, Mic, Loader2 } from "lucide-react";
import { processPdf } from "@/lib/pdf-client";
import { useCredits } from "@/hooks/use-credits";

const VOICE_OPTIONS: { value: VoiceOption; label: string }[] = [
  { value: "nova", label: "Nova (Female)" },
  { value: "alloy", label: "Alloy (Male)" },
  { value: "fable", label: "Phoebe (Female)" },
  { value: "echo", label: "Adam (Male)" },
];

export default function RespondPage() {
  const router = useRouter();
  const { refreshBalance } = useCredits();

  const [inputMethod, setInputMethod] = useState<InputMethod>("Upload Files");
  const [responseMode, setResponseMode] = useState<DialogueMode>("Deeper");
  const [selectedVoice, setSelectedVoice] = useState<VoiceOption>("nova");
  const [files, setFiles] = useState<File[]>([]);
  const [topicText, setTopicText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");

  const [questions, setQuestions] = useState<string[] | null>(null);
  const [selectedQuestion, setSelectedQuestion] = useState<string | null>(null);
  const [showQuestionDialog, setShowQuestionDialog] = useState(false);
  const [extractedText, setExtractedText] = useState<string>("");

  const [responseItems, setResponseItems] = useState<DialogueItem[]>([]);
  const [learningNotes, setLearningNotes] = useState<LearningNotesType | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [sessionTitle, setSessionTitle] = useState<string>("");
  const [charactersCount, setCharactersCount] = useState(0);
  const [ttsCostHKD, setTtsCostHKD] = useState(0);
  const [usedOwnApiKey, setUsedOwnApiKey] = useState(false);
  const [creditsConsumed, setCreditsConsumed] = useState(0);
  const [accessCode, setAccessCode] = useState<string | null>(null);
  const [expiryDays, setExpiryDays] = useState<number | null>(null);
  const [generatedTimestamp, setGeneratedTimestamp] = useState<string | null>(null);

  const pendingGenerateRef = useRef<{
    text: string;
    question: string | null;
    method: InputMethod;
    mode: DialogueMode;
    voice: VoiceOption;
  } | null>(null);

  const handleGenerate = useCallback(async (
    text: string,
    question: string | null,
    method: InputMethod,
    mode: DialogueMode,
    voice: VoiceOption
  ) => {
    setIsGenerating(true);
    setProgress(5);
    setProgressLabel("Generating response and study notes...");

    let generationId: string | null = null;

    try {
      const res = await fetch("/api/respond", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          question,
          responseMode: mode,
        }),
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

      const data: RespondResponse = await res.json();
      generationId = data.generationId ?? null;
      setResponseItems(data.response);
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
      const totalLines = data.response.length;
      const audioChunks: ArrayBuffer[] = [];
      let completed = 0;

      try {
        const batchSize = 5;
        for (let i = 0; i < totalLines; i += batchSize) {
          const batch = data.response.slice(i, i + batchSize);
          const promises = batch.map(async (item) => {
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
        uploadForm.append("audio", blob, "response.mp3");
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
            sessionType: "response",
            dialogueMode: mode,
            inputMethod: method,
            inputText: data.extractedText,
            transcript: data.response,
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
          `Response generated! Total cost: HK$${data.ttsCostHKD.toFixed(2)}`
        );
      } else {
        toast.success(
          `Response generated! Credits consumed: ${data.creditsConsumed}`
        );
      }
    } catch (error: unknown) {
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
      setIsGenerating(false);
    }
  }, [refreshBalance]);

  const handlePrepare = useCallback(async () => {
    if (inputMethod === "Upload Files" && files.length === 0) {
      toast.error("Please upload at least one file.");
      return;
    }
    if (inputMethod === "Enter Topic" && !topicText.trim()) {
      toast.error("Please enter a question.");
      return;
    }

    setIsPreparing(true);
    try {
      const formData = new FormData();
      formData.append("inputMethod", inputMethod);
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
          for (const pdfFile of pdfFiles) {
            try {
              const result = await processPdf(pdfFile);
              formData.append("fileName", result.fileName);
              if (result.text) {
                formData.append("preExtractedText", result.text);
              } else {
                result.images.forEach((img) => formData.append("files", img));
              }
            } catch {
              formData.append("files", pdfFile);
            }
          }
        }
      }

      const res = await fetch("/api/respond/prepare", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Failed to process input.");
      }

      const data = await res.json();
      setExtractedText(data.extractedText);

      if (data.questions && data.questions.length > 1) {
        setQuestions(data.questions);
        setSelectedQuestion(data.questions[0]);
        setShowQuestionDialog(true);
        pendingGenerateRef.current = {
          text: data.extractedText,
          question: null,
          method: inputMethod,
          mode: responseMode,
          voice: selectedVoice,
        };
      } else {
        setQuestions(null);
        setSelectedQuestion(null);
        handleGenerate(data.extractedText, null, inputMethod, responseMode, selectedVoice);
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : "An error occurred.";
      toast.error(message);
    } finally {
      setIsPreparing(false);
    }
  }, [inputMethod, responseMode, selectedVoice, files, topicText, handleGenerate]);

  const handleQuestionConfirm = useCallback(() => {
    setShowQuestionDialog(false);
    if (selectedQuestion && extractedText) {
      handleGenerate(extractedText, selectedQuestion, inputMethod, responseMode, selectedVoice);
    }
  }, [selectedQuestion, extractedText, inputMethod, responseMode, selectedVoice, handleGenerate]);

  const handleExportDocx = useCallback(async () => {
    if (!responseItems.length || !learningNotes) return;
    try {
      const res = await fetch("/api/export/docx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: responseItems,
          learningNotes,
          title: sessionTitle,
          extractedText,
          accessCode,
          sessionType: "response",
        }),
      });
      if (!res.ok) throw new Error("Export failed.");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Mr.NG-DiscussAI-response-${generatedTimestamp || "unknown"}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Document downloaded!");
    } catch {
      toast.error("Failed to export document.");
    }
  }, [responseItems, learningNotes, sessionTitle, extractedText, accessCode, generatedTimestamp]);

  const audioDownloadName = generatedTimestamp
    ? `Mr.NG-DiscussAI-response-${generatedTimestamp}.mp3`
    : undefined;

  const isBusy = isGenerating || isPreparing;

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mic className="size-6" /> Individual Response Question
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
                  <Label htmlFor="text">Enter Question</Label>
                </div>
              </RadioGroup>
            </div>

            {inputMethod === "Upload Files" ? (
              <FileUpload files={files} onFilesChange={setFiles} />
            ) : (
              <Textarea
                placeholder="Paste or type your question here..."
                value={topicText}
                onChange={(e) => setTopicText(e.target.value)}
                rows={6}
              />
            )}

            <div>
              <Label className="mb-2 block">Voice</Label>
              <RadioGroup
                value={selectedVoice}
                onValueChange={(v) => setSelectedVoice(v as VoiceOption)}
                className="flex flex-wrap gap-3"
              >
                {VOICE_OPTIONS.map((opt) => (
                  <div key={opt.value} className="flex items-center space-x-2">
                    <RadioGroupItem value={opt.value} id={`voice-${opt.value}`} />
                    <Label htmlFor={`voice-${opt.value}`}>{opt.label}</Label>
                  </div>
                ))}
              </RadioGroup>
            </div>

            <div>
              <Label className="mb-2 block">Depth of Response</Label>
              <RadioGroup
                value={responseMode}
                onValueChange={(v) => setResponseMode(v as DialogueMode)}
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="Deeper" id="deeper" disabled />
                  <Label htmlFor="deeper">
                    Deeper
                    <span className="text-xs text-muted-foreground ml-1">
                      (More elaborate vocabulary)
                    </span>
                  </Label>
                </div>
              </RadioGroup>
            </div>

            {isBusy && (
              <div className="space-y-2">
                <Progress value={progress} />
                <p className="text-sm text-muted-foreground text-center">
                  {progressLabel}
                </p>
              </div>
            )}

            <Button
              onClick={() => handlePrepare()}
              disabled={isBusy}
              size="lg"
              className="w-full"
            >
              {isBusy ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {isPreparing ? "Processing..." : "Generating..."}
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate Response
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {responseItems.length > 0 && (
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

            <TranscriptDisplay items={responseItems} />

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

      <Dialog open={showQuestionDialog} onOpenChange={setShowQuestionDialog}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Multiple Questions Detected</DialogTitle>
            <DialogDescription>
              We found multiple questions in your input. Please select which question you want to generate a response for.
            </DialogDescription>
          </DialogHeader>
          <RadioGroup
            value={selectedQuestion || ""}
            onValueChange={setSelectedQuestion}
            className="space-y-3 max-h-[50vh] overflow-y-auto"
          >
            {questions?.map((q, i) => (
              <div key={i} className="flex items-start space-x-2">
                <RadioGroupItem value={q} id={`q-${i}`} className="mt-1" />
                <Label htmlFor={`q-${i}`} className="text-sm leading-relaxed cursor-pointer">
                  {q}
                </Label>
              </div>
            ))}
          </RadioGroup>
          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline" onClick={() => setShowQuestionDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleQuestionConfirm} disabled={!selectedQuestion}>
              <Sparkles className="mr-2 h-4 w-4" />
              Generate Response
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
