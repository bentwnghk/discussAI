import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { isAdminEmail } from "@/lib/admin";
import { db } from "@/lib/db";
import { discussionSessions, users } from "@/lib/db/schema";
import { desc, asc, sql, ilike, or } from "drizzle-orm";
import { getGenerationCost } from "@/lib/db/credits";

export async function GET(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id || !isAdminEmail(session.user.email)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const { searchParams } = new URL(req.url);
    const sortBy = searchParams.get("sortBy") || "createdAt";
    const sortOrder = searchParams.get("sortOrder") || "desc";
    const search = searchParams.get("q") || "";

    const query = db
      .select({
        id: discussionSessions.id,
        userName: users.name,
        email: users.email,
        title: discussionSessions.title,
        dialogueMode: discussionSessions.dialogueMode,
        createdAt: discussionSessions.createdAt,
        usedOwnApiKey: discussionSessions.usedOwnApiKey,
        ttsCostHKD: discussionSessions.ttsCostHKD,
      })
      .from(discussionSessions)
      .innerJoin(users, sql`${discussionSessions.userId} = ${users.id}`);

    const conditions = search.trim()
      ? [
          ilike(users.name, `%${search.trim()}%`),
          ilike(users.email, `%${search.trim()}%`),
          ilike(discussionSessions.title, `%${search.trim()}%`),
          ilike(discussionSessions.dialogueMode, `%${search.trim()}%`),
        ]
      : [];

    const orderColumn =
      sortBy === "userName"
        ? users.name
        : sortBy === "title"
          ? discussionSessions.title
          : sortBy === "dialogueMode"
            ? discussionSessions.dialogueMode
            : discussionSessions.createdAt;

    const orderFn = sortOrder === "asc" ? asc : desc;

    const rows = conditions.length
      ? await query
          .where(or(...conditions))
          .orderBy(orderFn(orderColumn))
      : await query.orderBy(orderFn(orderColumn));

    return NextResponse.json({ discussions: rows, generationCost: getGenerationCost() });
  } catch (error) {
    console.error("Admin discussions GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch discussions." },
      { status: 500 }
    );
  }
}
