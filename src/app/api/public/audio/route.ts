import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { readFile, stat } from "fs/promises";
import path from "path";
import { AUDIO_TTL_MS } from "@/lib/audio-ttl";

const AUDIO_DIR = path.join(process.cwd(), "tmp", "audio");

export async function GET(req: NextRequest) {
  try {
    const code = req.nextUrl.searchParams.get("code");
    if (!code) {
      return NextResponse.json({ error: "Access code is required." }, { status: 400 });
    }

    const [result] = await db
      .select({ audioUrl: discussionSessions.audioUrl, createdAt: discussionSessions.createdAt })
      .from(discussionSessions)
      .where(eq(discussionSessions.accessCode, code.toUpperCase()))
      .limit(1);

    if (!result?.audioUrl) {
      return NextResponse.json({ error: "Invalid access code or no audio." }, { status: 404 });
    }

    if (Date.now() - new Date(result.createdAt).getTime() > AUDIO_TTL_MS) {
      return NextResponse.json({ error: "Audio has expired." }, { status: 410 });
    }

    const filename = result.audioUrl.split("/").pop();
    if (!filename || filename.includes("..")) {
      return new NextResponse("Forbidden", { status: 403 });
    }

    const filePath = path.join(AUDIO_DIR, filename);
    const data = await readFile(filePath);

    return new NextResponse(data, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": data.length.toString(),
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch {
    return new NextResponse("Not Found", { status: 404 });
  }
}
