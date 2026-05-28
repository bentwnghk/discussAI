import { NextRequest } from "next/server";
import { generateBrainstorm, generateDialogueFromBrainstorm, generateDialogueLearningNotes } from "@/lib/ai/dialogue-generator";
import { extractPartAText } from "@/lib/ai/prompts";
import { extractTextFromFile } from "@/lib/file-processing";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";
import { deductCredits, refundCredits, refundGeneration, getGenerationCost } from "@/lib/db/credits";
import { mkdir, writeFile } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

export const maxDuration = 180;

function sendEvent(controller: ReadableStreamDefaultController, event: string, data: unknown) {
  controller.enqueue(new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
}

export async function POST(req: NextRequest) {
  let userId: string | undefined;
  let generationId: string | undefined;
  let creditsDeducted = false;

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const session = await auth();
        if (!session?.user?.id) {
          sendEvent(controller, "error", { error: "Unauthorized" });
          controller.close();
          return;
        }
        userId = session.user.id;

        const apiKey = await getUserApiKey(session.user.id);
        const usedOwnApiKey = !!apiKey;

        if (!usedOwnApiKey) {
          const generationCost = getGenerationCost();
          const deduction = await deductCredits(
            userId,
            generationCost,
            "Discussion generation"
          );
          if (!deduction.success) {
            sendEvent(controller, "error", {
              error: deduction.error,
              creditsNeeded: generationCost,
              currentBalance: deduction.balance,
            });
            controller.close();
            return;
          }
          generationId = deduction.transactionId!;
          creditsDeducted = true;
        }

        const formData = await req.formData();
        const inputMethod = formData.get("inputMethod") as string;
        const dialogueMode = (formData.get("dialogueMode") as string) || "Normal";
        const textInput = (formData.get("text") as string) || "";

        let fullText = "";
        let topicLabel = "";

        if (inputMethod === "Upload Files") {
          const files: File[] = [];
          formData.getAll("files").forEach((f) => {
            if (f instanceof File) files.push(f);
          });
          const preExtractedTexts = formData.getAll("preExtractedText") as string[];
          const fileNames = formData.getAll("fileName") as string[];

          if (files.length === 0 && preExtractedTexts.length === 0) {
            if (creditsDeducted) {
              await refundCredits(userId, getGenerationCost(), "Refund: no files uploaded");
            }
            sendEvent(controller, "error", { error: "Please upload at least one file." });
            controller.close();
            return;
          }

          const allFileNames = [
            ...fileNames.map((n) => n.replace(/\.[^/.]+$/, "")),
          ];
          if (allFileNames.length === 0) {
            allFileNames.push(
              ...files.map((f) => f.name.replace(/\.[^/.]+$/, ""))
            );
          }
          topicLabel = allFileNames.join(", ");
          const texts: string[] = [];
          const tmpDir = path.join(process.cwd(), "tmp", "uploads");
          await mkdir(tmpDir, { recursive: true });

          for (const text of preExtractedTexts) {
            if (text.trim()) texts.push(text.trim());
          }

          for (const file of files) {
            const bytes = await file.arrayBuffer();
            const tmpPath = path.join(tmpDir, `${randomUUID()}-${file.name}`);
            await writeFile(tmpPath, Buffer.from(bytes));
            try {
              const text = await extractTextFromFile(tmpPath, apiKey);
              if (text.trim()) texts.push(text);
            } finally {
              const { unlink } = await import("fs/promises");
              await unlink(tmpPath).catch(() => {});
            }
          }

          fullText = texts.join("\n\n");
        } else if (inputMethod === "Enter Topic") {
          if (!textInput.trim()) {
            if (creditsDeducted) {
              await refundCredits(userId, getGenerationCost(), "Refund: empty topic");
            }
            sendEvent(controller, "error", { error: "Please enter a topic." });
            controller.close();
            return;
          }
          fullText = textInput;
          topicLabel = textInput.trim().slice(0, 80);
          if (textInput.trim().length > 80) topicLabel += "…";
        } else {
          if (creditsDeducted) {
            await refundCredits(userId, getGenerationCost(), "Refund: invalid input method");
          }
          sendEvent(controller, "error", { error: "Invalid input method." });
          controller.close();
          return;
        }

        if (!fullText.trim()) {
          if (creditsDeducted) {
            await refundCredits(userId, getGenerationCost(), "Refund: no text content");
          }
          sendEvent(controller, "error", { error: "No text content to process." });
          controller.close();
          return;
        }

        const mode = dialogueMode as "Normal" | "Deeper";

        sendEvent(controller, "progress", { step: "brainstorm", label: "Step 1/3: Brainstorming ideas...", progress: 20 });
        const brainstorm = await generateBrainstorm(fullText, mode, apiKey);
        const brainstormText = `Topic: ${brainstorm.topic_summary}\n\nQuestion Prompts:\n${brainstorm.question_prompts.map((q, i) => `${i + 1}. ${q}`).join("\n")}\n\nKey Points:\n${brainstorm.key_points.map((p, i) => `${i + 1}. ${p}`).join("\n")}\n\nBrainstorm:\n${brainstorm.scratchpad}`;

        sendEvent(controller, "progress", { step: "dialogue", label: "Step 2/3: Generating dialogue...", progress: 35 });
        const dialogueResult = await generateDialogueFromBrainstorm(brainstormText, mode, apiKey);

        sendEvent(controller, "progress", { step: "learning_notes", label: "Step 3/3: Creating study notes...", progress: 50 });
        const dialogueTextForNotes = dialogueResult.dialogue.map((item: { speaker: string; text: string }) => `${item.speaker}: ${item.text}`).join("\n");
        const learningNotesResult = await generateDialogueLearningNotes(dialogueTextForNotes, brainstormText, mode, apiKey);

        const charactersCount = dialogueResult.dialogue.reduce(
          (sum: number, item: { text: string }) => sum + item.text.length,
          0
        );
        const ttsCostHKD = (charactersCount / 1_000_000) * 15 * 7.8;

        const title = `Group Discussion - ${topicLabel}`;

        sendEvent(controller, "result", {
          dialogue: dialogueResult.dialogue,
          learningNotes: learningNotesResult.learning_notes,
          scratchpad: brainstorm.scratchpad,
          charactersCount,
          ttsCostHKD,
          title,
          extractedText: extractPartAText(fullText),
          generationId: generationId || null,
          usedOwnApiKey,
          creditsConsumed: creditsDeducted ? getGenerationCost() : 0,
        });
        controller.close();
      } catch (error: unknown) {
        const message =
          error instanceof Error ? error.message : "An error occurred.";
        console.error("Generate error:", error);
        if (userId && creditsDeducted && generationId) {
          await refundGeneration(userId, generationId).catch(() => {});
        } else if (userId && creditsDeducted) {
          await refundCredits(
            userId,
            getGenerationCost(),
            "Refund: generation failed"
          ).catch(() => {});
        }
        sendEvent(controller, "error", { error: message });
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
