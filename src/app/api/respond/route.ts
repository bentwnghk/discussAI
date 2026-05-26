import { NextRequest, NextResponse } from "next/server";
import { generateIndividualResponse } from "@/lib/ai/dialogue-generator";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";
import { deductCredits, refundCredits, refundGeneration, getResponseCost } from "@/lib/db/credits";

export const maxDuration = 180;

export async function POST(req: NextRequest) {
  let userId: string | undefined;
  let generationId: string | undefined;
  let creditsDeducted = false;
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    userId = session.user.id;

    const apiKey = await getUserApiKey(session.user.id);
    const usedOwnApiKey = !!apiKey;

    if (!usedOwnApiKey) {
      const responseCost = getResponseCost();
      const deduction = await deductCredits(
        userId,
        responseCost,
        "Individual response generation"
      );
      if (!deduction.success) {
        return NextResponse.json(
          {
            error: deduction.error,
            creditsNeeded: responseCost,
            currentBalance: deduction.balance,
          },
          { status: 402 }
        );
      }
      generationId = deduction.transactionId!;
      creditsDeducted = true;
    }

    const body = await req.json();
    const text = (body.text as string) || "";
    const question = (body.question as string) || "";
    const responseMode = (body.responseMode as string) || "Normal";

    if (!text.trim()) {
      if (creditsDeducted) {
        await refundCredits(userId, getResponseCost(), "Refund: no text content");
      }
      return NextResponse.json(
        { error: "No text content to process." },
        { status: 400 }
      );
    }

    const promptText = question
      ? `Question:\n${question}\n\nSource material:\n${text}`
      : text;

    const mode = responseMode as "Normal" | "Deeper";
    const result = await generateIndividualResponse(promptText, mode, apiKey);

    const charactersCount = result.response.reduce(
      (sum: number, item: { text: string }) => sum + item.text.length,
      0
    );
    const ttsCostHKD = (charactersCount / 1_000_000) * 15 * 7.8;

    const topicLabel = question
      ? question.trim().slice(0, 80) + (question.trim().length > 80 ? "…" : "")
      : text.trim().slice(0, 80) + (text.trim().length > 80 ? "…" : "");

    const title = `Individual Response - ${topicLabel}`;

    return NextResponse.json({
      response: result.response,
      learningNotes: result.learning_notes,
      scratchpad: result.scratchpad,
      charactersCount,
      ttsCostHKD,
      title,
      extractedText: question || text,
      generationId: generationId || null,
      usedOwnApiKey,
      creditsConsumed: creditsDeducted ? getResponseCost() : 0,
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "An error occurred.";
    console.error("Respond error:", error);
    if (userId && creditsDeducted && generationId) {
      await refundGeneration(userId, generationId).catch(() => {});
    } else if (userId && creditsDeducted) {
      await refundCredits(
        userId,
        getResponseCost(),
        "Refund: response generation failed"
      ).catch(() => {});
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
