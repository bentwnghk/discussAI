import type { Speaker } from "@/types";

const SPEAKER_ORDER: Speaker[] = [
  "Candidate A",
  "Candidate B",
  "Candidate C",
  "Candidate D",
];

const DEFAULT_TTS_MODEL = "tts-1";
const DEFAULT_TTS_VOICES = ["nova", "alloy", "fable", "echo"];

export function getTTSModel(): string {
  return process.env.TTS_MODEL?.trim() || DEFAULT_TTS_MODEL;
}

export function getAvailableVoices(): string[] {
  const raw = process.env.TTS_VOICES;
  if (!raw?.trim()) return [...DEFAULT_TTS_VOICES];

  const parsed = raw
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);

  return SPEAKER_ORDER.map((_, i) => parsed[i] ?? DEFAULT_TTS_VOICES[i]);
}

export function getVoiceForSpeaker(speaker: Speaker): string {
  const voices = getAvailableVoices();
  const index = SPEAKER_ORDER.indexOf(speaker);
  return voices[index] ?? voices[0];
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
      model: getTTSModel(),
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
