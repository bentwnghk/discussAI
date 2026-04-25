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

export default function DiscussPage() {
  const router = useRouter();
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
    setProgress(10);
    setProgressLabel("Analyzing input...");

    try {
      const formData = new FormData();
      formData.append("inputMethod", inputMethod);
      formData.append("dialogueMode", dialogueMode);
      if (inputMethod === "Enter Topic") {
        formData.append("text", topicText);
      }
      if (inputMethod === "Upload Files") {
        files.forEach((f) => formData.append("files", f));
      }

      setProgressLabel("Generating dialogue and study notes...");
      setProgress(20);

      const res = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Generation failed.");
      }

      const data: GenerateResponse = await res.json();
      setDialogueItems(data.dialogue);
      setLearningNotes(data.learningNotes);
      setSessionTitle(data.title);
      setCharactersCount(data.charactersCount);
      setTtsCostHKD(data.ttsCostHKD);
      setExtractedText(data.extractedText);

      setProgressLabel("Generating audio...");
      setProgress(40);

      const totalLines = data.dialogue.length;
      const audioChunks: ArrayBuffer[] = [];
      let completed = 0;

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
          }),
        });
      } catch {
        toast.warning("Session saved locally but could not sync to server.");
      }

      setProgress(100);
      setProgressLabel("Done!");
      toast.success(
        `Discussion generated! Total cost: HK$${data.ttsCostHKD.toFixed(2)}`
      );
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "An error occurred.";
      toast.error(message);
    } finally {
      setIsGenerating(false);
    }
  }, [inputMethod, dialogueMode, files, topicText]);

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
      a.download = "discussion-notes.docx";
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
            <CardTitle className="flex items-center gap-2">
              <span className="text-2xl">Discussion Topic</span>
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
                      (more elaboration)
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
                  Generate Discussion with Audio and Study Notes
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
                  <CardTitle>Task 任務</CardTitle>
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
                Cost: HK${ttsCostHKD.toFixed(2)} | {charactersCount} characters
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
