import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq, and } from "drizzle-orm";
import { unlink } from "fs/promises";
import path from "path";

const AUDIO_DIR = path.join(process.cwd(), "tmp", "audio");

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = await params;
    const [result] = await db
      .select()
      .from(discussionSessions)
      .where(
        and(
          eq(discussionSessions.id, id),
          eq(discussionSessions.userId, session.user.id)
        )
      );

    if (!result) {
      return NextResponse.json({ error: "Not found." }, { status: 404 });
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("History GET [id] error:", error);
    return NextResponse.json(
      { error: "Failed to fetch session." },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = await params;
    const body = await req.json();
    const { title } = body;

    if (!title || typeof title !== "string" || !title.trim()) {
      return NextResponse.json({ error: "Title is required." }, { status: 400 });
    }

    const [updated] = await db
      .update(discussionSessions)
      .set({ title: title.trim() })
      .where(
        and(
          eq(discussionSessions.id, id),
          eq(discussionSessions.userId, session.user.id)
        )
      )
      .returning();

    if (!updated) {
      return NextResponse.json({ error: "Not found." }, { status: 404 });
    }

    return NextResponse.json(updated);
  } catch (error) {
    console.error("History PATCH error:", error);
    return NextResponse.json(
      { error: "Failed to rename session." },
      { status: 500 }
    );
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = await params;

    const [existing] = await db
      .select({ audioUrl: discussionSessions.audioUrl })
      .from(discussionSessions)
      .where(
        and(
          eq(discussionSessions.id, id),
          eq(discussionSessions.userId, session.user.id)
        )
      );

    if (existing?.audioUrl) {
      const filename = existing.audioUrl.split("/").pop();
      if (filename && !filename.includes("..")) {
        await unlink(path.join(AUDIO_DIR, filename)).catch(() => {});
      }
    }

    await db
      .delete(discussionSessions)
      .where(
        and(
          eq(discussionSessions.id, id),
          eq(discussionSessions.userId, session.user.id)
        )
      );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("History DELETE error:", error);
    return NextResponse.json(
      { error: "Failed to delete session." },
      { status: 500 }
    );
  }
}
