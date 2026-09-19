import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { generateTTSAudio, getAvailableVoices, getTTSModel, getVoiceForSpeaker } from "@/lib/tts/generate";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";
import type { Speaker } from "@/types";

const ttsRequestSchema = z
  .object({
    text: z.string().min(1),
    speaker: z.string().optional(),
    voice: z.string().optional(),
  })
  .refine((data) => data.speaker || data.voice, {
    message: "Either speaker or voice is required.",
  });

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({
    model: getTTSModel(),
    voices: getAvailableVoices(),
  });
}

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const apiKey = await getUserApiKey(session.user.id);

    const parsed = ttsRequestSchema.safeParse(await req.json());
    if (!parsed.success) {
      return NextResponse.json(
        { error: parsed.error.issues[0]?.message ?? "Invalid request." },
        { status: 400 }
      );
    }

    const { text, speaker, voice: explicitVoice } = parsed.data;
    const voice = explicitVoice || getVoiceForSpeaker(speaker as Speaker);

    const audioBuffer = await generateTTSAudio(text, voice, apiKey);

    return new NextResponse(new Uint8Array(audioBuffer), {
      headers: {
        "Content-Type": "audio/mpeg",
        "Content-Length": audioBuffer.length.toString(),
      },
    });
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : "TTS generation failed.";
    console.error("TTS error:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
