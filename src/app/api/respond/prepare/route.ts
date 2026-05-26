import { NextRequest, NextResponse } from "next/server";
import { extractTextFromFile } from "@/lib/file-processing";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";
import { extractQuestions } from "@/lib/ai/dialogue-generator";
import { mkdir, writeFile } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const apiKey = await getUserApiKey(session.user.id);

    const formData = await req.formData();
    const inputMethod = formData.get("inputMethod") as string;
    const textInput = (formData.get("text") as string) || "";

    let fullText = "";

    if (inputMethod === "Upload Files") {
      const files: File[] = [];
      formData.getAll("files").forEach((f) => {
        if (f instanceof File) files.push(f);
      });
      const preExtractedTexts = formData.getAll("preExtractedText") as string[];

      if (files.length === 0 && preExtractedTexts.length === 0) {
        return NextResponse.json(
          { error: "Please upload at least one file." },
          { status: 400 }
        );
      }

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
        return NextResponse.json(
          { error: "Please enter a question." },
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

    const questions = await extractQuestions(fullText, apiKey);

    return NextResponse.json({
      questions: questions.length > 0 ? questions : null,
      extractedText: fullText,
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "An error occurred.";
    console.error("Prepare error:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
