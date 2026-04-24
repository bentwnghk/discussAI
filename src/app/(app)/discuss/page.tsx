"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
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
import { getVoiceForSpeaker } from "@/lib/tts/generate";

const API_KEY_STORAGE_KEY = "discussai_api_key";

export default function DiscussPage() {
  const router = useRouter();
  const [inputMethod, setInputMethod] = useState<InputMethod>("Upload Files");
  const [dialogueMode, setDialogueMode] = useState<DialogueMode>("Normal");
  const [files, setFiles] = useState<File[]>([]);
  const [topicText, setTopicText] = useState("");
  const [apiKey, setApiKey] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(API_KEY_STORAGE_KEY) || "";
    }
    return "";
  });
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

  const handleApiKeyChange = useCallback((value: string) => {
    setApiKey(value);
    localStorage.setItem(API_KEY_STORAGE_KEY, value);
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
    setProgress(10);
    setProgressLabel("Analyzing input...");

    try {
      const formData = new FormData();
      formData.append("inputMethod", inputMethod);
      formData.append("dialogueMode", dialogueMode);
      if (apiKey) formData.append("apiKey", apiKey);
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
              apiKey: apiKey || undefined,
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
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      setProgressLabel("Saving session...");
      setProgress(95);

      const reader = new FileReader();
      reader.onload = async () => {
        const base64Audio = (reader.result as string).split(",")[1];
        try {
          await fetch("/api/history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: data.title,
              dialogueMode: data.dialogue
                ? dialogueMode
                : "Normal",
              inputMethod,
              inputText:
                inputMethod === "Enter Topic" ? topicText : null,
              transcript: data.dialogue,
              learningNotes: data.learningNotes,
              audioUrl: base64Audio,
              charactersCount: data.charactersCount,
              ttsCostHKD: data.ttsCostHKD,
            }),
          });
        } catch {
          toast.warning("Session saved locally but could not sync to server.");
        }
      };
      reader.readAsDataURL(blob);

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
  }, [inputMethod, dialogueMode, files, topicText, apiKey]);

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

            <Accordion>
              <AccordionItem value="advanced">
                <AccordionTrigger className="text-sm">
                  Advanced Settings
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3 pt-2">
                    <p className="text-sm text-muted-foreground">
                      Get your API key{" "}
                      <a
                        href="https://api.mr5ai.com"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline"
                      >
                        here
                      </a>
                    </p>
                    <Input
                      type="password"
                      placeholder="API Key (optional if set server-side)"
                      value={apiKey}
                      onChange={(e) => handleApiKeyChange(e.target.value)}
                    />
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>

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
              {isGenerating
                ? "Generating..."
                : "Generate Discussion with Audio and Study Notes"}
            </Button>
          </CardContent>
        </Card>

        {dialogueItems.length > 0 && (
          <>
            <Separator />

            <AudioPlayer src={audioUrl} />

            <TranscriptDisplay items={dialogueItems} />

            {learningNotes && <LearningNotes notes={learningNotes} />}

            <div className="flex justify-between items-center">
              <p className="text-sm text-muted-foreground">
                Cost: HK${ttsCostHKD.toFixed(2)} | {charactersCount} characters
              </p>
              <Button onClick={handleExportDocx} variant="outline">
                Download as Word Document
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
            View Practice History
          </Button>
        </div>
      </div>
    </div>
  );
}
