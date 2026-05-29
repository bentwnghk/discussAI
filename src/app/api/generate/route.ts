import { NextRequest, NextResponse } from "next/server";
import { generateDialogue } from "@/lib/ai/dialogue-generator";
import { extractPartAText } from "@/lib/ai/prompts";
import { extractTextFromFile } from "@/lib/file-processing";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";
import { deductCredits, refundCredits, refundGeneration, getGenerationCost } from "@/lib/db/credits";
import { mkdir, writeFile } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

export const maxDuration = 240;

type ProgressEvent = {
  type: "progress";
  stage: string;
  message: string;
};

type CompleteEvent = {
  type: "complete";
  data: Record<string, unknown>;
};

type ErrorEvent = {
  type: "error";
  message: string;
};

type StreamEvent = ProgressEvent | CompleteEvent | ErrorEvent;

function sendEvent(controller: ReadableStreamDefaultController, event: StreamEvent) {
  const text = `data: ${JSON.stringify(event)}\n\n`;
  controller.enqueue(new TextEncoder().encode(text));
}

export async function POST(req: NextRequest) {
  let generationId: string | undefined;
  let creditsDeducted = false;

  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const userId = session.user.id;

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
      return NextResponse.json(
        {
          error: deduction.error,
          creditsNeeded: generationCost,
          currentBalance: deduction.balance,
        },
        { status: 402 }
      );
    }
    generationId = deduction.transactionId!;
    creditsDeducted = true;
  }

  const formData = await req.formData();
  const inputMethod = formData.get("inputMethod") as string;
  const dialogueMode = (formData.get("dialogueMode") as string) || "Normal";
  const textInput = (formData.get("text") as string) || "";

  const stream = new ReadableStream({
    async start(controller) {
      try {
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
              await refundCredits(userId!, getGenerationCost(), "Refund: no files uploaded");
            }
            sendEvent(controller, { type: "error", message: "Please upload at least one file." });
            controller.close();
            return;
          }

          sendEvent(controller, {
            type: "progress",
            stage: "processing_files",
            message: "Extracting text from uploaded files...",
          });

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

          const totalFiles = files.length;
          for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (totalFiles > 1) {
              sendEvent(controller, {
                type: "progress",
                stage: "processing_file",
                message: `Extracting text from file ${i + 1}/${totalFiles}...`,
              });
            }
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
              await refundCredits(userId!, getGenerationCost(), "Refund: empty topic");
            }
            sendEvent(controller, { type: "error", message: "Please enter a topic." });
            controller.close();
            return;
          }
          fullText = textInput;
          topicLabel = textInput.trim().slice(0, 80);
          if (textInput.trim().length > 80) topicLabel += "…";
        } else {
          if (creditsDeducted) {
            await refundCredits(userId!, getGenerationCost(), "Refund: invalid input method");
          }
          sendEvent(controller, { type: "error", message: "Invalid input method." });
          controller.close();
          return;
        }

        if (!fullText.trim()) {
          if (creditsDeducted) {
            await refundCredits(userId!, getGenerationCost(), "Refund: no text content");
          }
          sendEvent(controller, { type: "error", message: "No text content to process." });
          controller.close();
          return;
        }

        sendEvent(controller, {
          type: "progress",
          stage: "generating",
          message: "AI is generating discussion transcript and study notes...",
        });

        const mode = dialogueMode as "Normal" | "Deeper";
        const dialogue = await generateDialogue(fullText, mode, apiKey);

        sendEvent(controller, {
          type: "progress",
          stage: "finalizing",
          message: "Finalizing response...",
        });

        const charactersCount = dialogue.dialogue.reduce(
          (sum, item) => sum + item.text.length,
          0
        );
        const ttsCostHKD = (charactersCount / 1_000_000) * 15 * 7.8;

        const title = `Group Discussion - ${topicLabel}`;

        sendEvent(controller, {
          type: "complete",
          data: {
            dialogue: dialogue.dialogue,
            learningNotes: dialogue.learning_notes,
            scratchpad: dialogue.scratchpad,
            charactersCount,
            ttsCostHKD,
            title,
            extractedText: extractPartAText(fullText),
            generationId: generationId || null,
            usedOwnApiKey,
            creditsConsumed: creditsDeducted ? getGenerationCost() : 0,
          },
        });
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
        sendEvent(controller, { type: "error", message });
      } finally {
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
