import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { mkdir, writeFile } from "fs/promises";
import path from "path";
import { randomUUID } from "crypto";

const AUDIO_DIR = path.join(process.cwd(), "tmp", "audio");

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await req.formData();
    const file = formData.get("audio") as File | null;
    if (!file) {
      return NextResponse.json({ error: "No audio file." }, { status: 400 });
    }

    await mkdir(AUDIO_DIR, { recursive: true });

    const filename = `${randomUUID()}.mp3`;
    const filePath = path.join(AUDIO_DIR, filename);
    const bytes = await file.arrayBuffer();
    await writeFile(filePath, Buffer.from(bytes));

    return NextResponse.json({ audioUrl: `/api/audio/${filename}` });
  } catch (error) {
    console.error("Audio upload error:", error);
    return NextResponse.json(
      { error: "Failed to save audio." },
      { status: 500 }
    );
  }
}
