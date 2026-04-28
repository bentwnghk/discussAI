import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { AUDIO_TTL_MS } from "@/lib/audio-ttl";

export async function GET(req: NextRequest) {
  try {
    const code = req.nextUrl.searchParams.get("code");
    if (!code) {
      return NextResponse.json({ error: "Access code is required." }, { status: 400 });
    }

    const [result] = await db
      .select({
        id: discussionSessions.id,
        title: discussionSessions.title,
        transcript: discussionSessions.transcript,
        audioUrl: discussionSessions.audioUrl,
        createdAt: discussionSessions.createdAt,
      })
      .from(discussionSessions)
      .where(eq(discussionSessions.accessCode, code.toUpperCase()))
      .limit(1);

    if (!result) {
      return NextResponse.json({ error: "Invalid access code." }, { status: 404 });
    }

    if (!result.audioUrl) {
      return NextResponse.json({ error: "No audio available for this session." }, { status: 404 });
    }

    const audioExpiresAt = new Date(
      new Date(result.createdAt).getTime() + AUDIO_TTL_MS
    ).toISOString();
    const isExpired = Date.now() > new Date(audioExpiresAt).getTime();

    return NextResponse.json({
      title: result.title,
      transcript: result.transcript,
      audioUrl: result.audioUrl,
      audioExpiresAt,
      audioExpired: isExpired,
    });
  } catch (error) {
    console.error("Public session GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch session." },
      { status: 500 }
    );
  }
}
