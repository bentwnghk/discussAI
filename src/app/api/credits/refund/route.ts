import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { refundGeneration } from "@/lib/db/credits";

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { generationId } = body;
  if (!generationId || typeof generationId !== "string") {
    return NextResponse.json(
      { error: "generationId is required" },
      { status: 400 }
    );
  }

  const result = await refundGeneration(session.user.id, generationId);

  if (result.refunded) {
    return NextResponse.json({ refunded: true });
  }
  return NextResponse.json({ refunded: false, reason: result.reason });
}
