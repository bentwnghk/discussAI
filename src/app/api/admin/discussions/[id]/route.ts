import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { getGenerationCost } from "@/lib/db/credits";
import { AUDIO_TTL_MS } from "@/lib/audio-ttl";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const session = await auth();
    if (!session?.user?.id || !isAdminEmail(session.user.email)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const { id } = await params;
    const [result] = await db
      .select()
      .from(discussionSessions)
      .where(eq(discussionSessions.id, id));

    if (!result) {
      return NextResponse.json({ error: "Not found." }, { status: 404 });
    }

    const audioExpiresAt = result.audioUrl
      ? new Date(
          new Date(result.createdAt).getTime() + AUDIO_TTL_MS
        ).toISOString()
      : null;

    return NextResponse.json({
      ...result,
      generationCost: getGenerationCost(),
      audioExpiresAt,
    });
  } catch (error) {
    console.error("Admin discussion detail GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch session." },
      { status: 500 }
    );
  }
}
