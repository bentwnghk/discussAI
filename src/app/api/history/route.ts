import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq, desc } from "drizzle-orm";
import { getGenerationCost } from "@/lib/db/credits";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const sessions = await db
      .select()
      .from(discussionSessions)
      .where(eq(discussionSessions.userId, session.user.id))
      .orderBy(desc(discussionSessions.createdAt));

    return NextResponse.json({ sessions, generationCost: getGenerationCost() });
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
        dialogueMode,
        inputMethod,
        inputText: inputText || null,
        transcript,
        learningNotes,
        audioUrl: audioUrl || null,
        charactersCount: charactersCount || 0,
        ttsCostHKD: ttsCostHKD || 0,
        usedOwnApiKey: !!usedOwnApiKey,
      })
      .returning();

    return NextResponse.json(newSession, { status: 201 });
  } catch (error) {
    console.error("History POST error:", error);
    return NextResponse.json(
      { error: "Failed to save session." },
      { status: 500 }
    );
  }
}
