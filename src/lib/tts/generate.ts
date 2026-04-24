import type { Speaker } from "@/types";

const OPENAI_VOICE_MAPPINGS: Record<Speaker, string> = {
  "Candidate A": "nova",
  "Candidate B": "alloy",
  "Candidate C": "fable",
  "Candidate D": "echo",
};

export function getVoiceForSpeaker(speaker: Speaker): string {
  return OPENAI_VOICE_MAPPINGS[speaker];
}

export async function generateTTSAudio(
  text: string,
  voice: string,
  apiKey?: string
): Promise<Buffer> {
  const effectiveApiKey = apiKey || process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL;

  if (!effectiveApiKey) throw new Error("API key not configured.");
  if (!baseUrl) throw new Error("Base URL not configured.");

  const endpoint = `${baseUrl.replace(/\/$/, "")}/audio/speech`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${effectiveApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "tts-1",
      voice,
      input: text,
      response_format: "mp3",
    }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new Error(
      `TTS request failed (${response.status}): ${errorText}`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}
