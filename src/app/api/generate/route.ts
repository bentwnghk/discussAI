import { NextRequest, NextResponse } from "next/server";
import { generateDialogue } from "@/lib/ai/dialogue-generator";
import { extractTextFromFile } from "@/lib/file-processing";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";
import { deductCredits, refundCredits, getGenerationCost } from "@/lib/db/credits";
import { mkdir, writeFile } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  let userId: string | undefined;
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    userId = session.user.id;

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

    const apiKey = await getUserApiKey(session.user.id);

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
        await refundCredits(userId, generationCost, "Refund: no files uploaded");
        return NextResponse.json(
          { error: "Please upload at least one file." },
          { status: 400 }
        );
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
        await refundCredits(userId, generationCost, "Refund: empty topic");
        return NextResponse.json(
          { error: "Please enter a topic." },
          { status: 400 }
        );
      }
      fullText = textInput;
      topicLabel = textInput.trim().slice(0, 80);
      if (textInput.trim().length > 80) topicLabel += "…";
    } else {
      await refundCredits(userId, generationCost, "Refund: invalid input method");
      return NextResponse.json(
        { error: "Invalid input method." },
        { status: 400 }
      );
    }

    if (!fullText.trim()) {
      await refundCredits(userId, generationCost, "Refund: no text content");
      return NextResponse.json(
        { error: "No text content to process." },
        { status: 400 }
      );
    }

    const mode = dialogueMode as "Normal" | "Deeper";
    const dialogue = await generateDialogue(fullText, mode, apiKey);

    const charactersCount = dialogue.dialogue.reduce(
      (sum, item) => sum + item.text.length,
      0
    );
    const ttsCostHKD = (charactersCount / 1_000_000) * 15 * 7.8;

    const title = `Group Discussion - ${topicLabel}`;

    return NextResponse.json({
      dialogue: dialogue.dialogue,
      learningNotes: dialogue.learning_notes,
      scratchpad: dialogue.scratchpad,
      charactersCount,
      ttsCostHKD,
      title,
      extractedText: fullText,
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "An error occurred.";
    console.error("Generate error:", error);
    if (userId) {
      await refundCredits(
        userId,
        getGenerationCost(),
        "Refund: generation failed"
      ).catch(() => {});
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
