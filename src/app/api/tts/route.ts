import { NextRequest, NextResponse } from "next/server";
import { generateTTSAudio } from "@/lib/tts/generate";
import { auth } from "@/lib/auth";
import { getUserApiKey } from "@/lib/db/user-api-key";

export async function POST(req: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const apiKey = await getUserApiKey(session.user.id);

    const body = await req.json();
    const { text, voice } = body as {
      text: string;
      voice: string;
    };

    if (!text || !voice) {
      return NextResponse.json(
        { error: "Text and voice are required." },
        { status: 400 }
      );
    }

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
