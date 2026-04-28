import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { refundGeneration, refundLastGeneration } from "@/lib/db/credits";

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { generationId, refundLast } = body as {
    generationId?: string;
    refundLast?: boolean;
  };

  let result: { refunded: boolean; reason?: string };

  if (generationId && typeof generationId === "string") {
    result = await refundGeneration(session.user.id, generationId);
  } else if (refundLast) {
    result = await refundLastGeneration(session.user.id);
  } else {
    return NextResponse.json(
      { error: "generationId or refundLast is required" },
      { status: 400 }
    );
  }

  if (result.refunded) {
    return NextResponse.json({ refunded: true });
  }
  return NextResponse.json({ refunded: false, reason: result.reason });
}
