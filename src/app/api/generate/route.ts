import { NextRequest, NextResponse } from "next/server";
import { generateDialogue } from "@/lib/ai/dialogue-generator";
import { extractTextFromFile } from "@/lib/file-processing";
import { auth } from "@/lib/auth";
import { mkdir, writeFile } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await req.formData();
    const inputMethod = formData.get("inputMethod") as string;
    const dialogueMode = (formData.get("dialogueMode") as string) || "Normal";
    const apiKey = (formData.get("apiKey") as string) || undefined;
    const textInput = (formData.get("text") as string) || "";

    let fullText = "";

    if (inputMethod === "Upload Files") {
      const files: File[] = [];
      formData.getAll("files").forEach((f) => {
        if (f instanceof File) files.push(f);
      });

      if (files.length === 0) {
        return NextResponse.json(
          { error: "Please upload at least one file." },
          { status: 400 }
        );
      }

      const texts: string[] = [];
      const tmpDir = path.join(process.cwd(), "tmp", "uploads");
      await mkdir(tmpDir, { recursive: true });

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
        return NextResponse.json(
          { error: "Please enter a topic." },
          { status: 400 }
        );
      }
      fullText = textInput;
    } else {
      return NextResponse.json(
        { error: "Invalid input method." },
        { status: 400 }
      );
    }

    if (!fullText.trim()) {
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

    const title = `Group Discussion - ${new Date().toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" })}`;

    return NextResponse.json({
      dialogue: dialogue.dialogue,
      learningNotes: dialogue.learning_notes,
      scratchpad: dialogue.scratchpad,
      charactersCount,
      ttsCostHKD,
      title,
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "An error occurred.";
    console.error("Generate error:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
