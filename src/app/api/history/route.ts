import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq, desc, and } from "drizzle-orm";
import { getGenerationCost, getResponseCost } from "@/lib/db/credits";
import { generateUniqueAccessCode } from "@/lib/db/access-code";
import { AUDIO_TTL_MS } from "@/lib/audio-ttl";

export async function GET(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const sessionType = searchParams.get("sessionType");

    const conditions = [eq(discussionSessions.userId, session.user.id)];
    if (sessionType && (sessionType === "discussion" || sessionType === "response")) {
      conditions.push(eq(discussionSessions.sessionType, sessionType));
    }

    const sessions = await db
      .select()
      .from(discussionSessions)
      .where(conditions.length === 1 ? conditions[0] : and(...conditions))
      .orderBy(desc(discussionSessions.createdAt));

    return NextResponse.json({ sessions, generationCost: getGenerationCost(), responseCost: getResponseCost() });
  } catch (error) {
    console.error("History GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch history." },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      title,
      sessionType,
      dialogueMode,
      inputMethod,
      inputText,
      transcript,
      learningNotes,
      audioUrl,
      charactersCount,
      ttsCostHKD,
      usedOwnApiKey,
    } = body;

    const [newSession] = await db
      .insert(discussionSessions)
      .values({
        userId: session.user.id,
        title,
        sessionType: sessionType || "discussion",
        dialogueMode,
        inputMethod,
        inputText: inputText || null,
        transcript,
        learningNotes,
        audioUrl: audioUrl || null,
        accessCode: await generateUniqueAccessCode(),
        charactersCount: charactersCount || 0,
        ttsCostHKD: ttsCostHKD || 0,
        usedOwnApiKey: !!usedOwnApiKey,
      })
      .returning();

    const audioExpiresAt = newSession.audioUrl
      ? new Date(new Date(newSession.createdAt).getTime() + AUDIO_TTL_MS).toISOString()
      : null;

    return NextResponse.json({ ...newSession, audioExpiresAt }, { status: 201 });
  } catch (error) {
    console.error("History POST error:", error);
    return NextResponse.json(
      { error: "Failed to save session." },
      { status: 500 }
    );
  }
}
