import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";
import { eq } from "drizzle-orm";

export async function GET() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const [user] = await db
      .select({ apiKey: users.apiKey })
      .from(users)
      .where(eq(users.id, session.user.id));

    return NextResponse.json({ apiKey: user?.apiKey || "" });
  } catch (error) {
    console.error("API key GET error:", error);
    return NextResponse.json(
      { error: "Failed to fetch API key." },
      { status: 500 }
    );
  }
}

export async function PUT(req: Request) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { apiKey } = await req.json();

    await db
      .update(users)
      .set({ apiKey: apiKey || null })
      .where(eq(users.id, session.user.id));

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("API key PUT error:", error);
    return NextResponse.json(
      { error: "Failed to save API key." },
      { status: 500 }
    );
  }
}
