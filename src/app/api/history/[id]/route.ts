import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq, and } from "drizzle-orm";

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
