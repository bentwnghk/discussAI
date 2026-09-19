import { Mp3Encoder } from "@breezystack/lamejs";
import type { Speaker } from "@/types";

const SPEAKER_ORDER: Speaker[] = [
  "Candidate A",
  "Candidate B",
  "Candidate C",
  "Candidate D",
];

const DEFAULT_TTS_MODEL = "tts-1";
const DEFAULT_TTS_VOICES = ["nova", "alloy", "fable", "echo"];
const DEFAULT_TTS_RESPONSE_FORMAT = "mp3";
const DEFAULT_PCM_SAMPLE_RATE = 24000;

export function getTTSModel(): string {
  return process.env.TTS_MODEL?.trim() || DEFAULT_TTS_MODEL;
}

function getTTSResponseFormat(): string {
  const format = process.env.TTS_RESPONSE_FORMAT?.trim() || DEFAULT_TTS_RESPONSE_FORMAT;
  if (format !== "mp3" && format !== "pcm") {
    throw new Error(
      `Unsupported TTS_RESPONSE_FORMAT "${format}". Supported values: mp3, pcm.`
    );
  }
  return format;
}

function getPCMSampleRate(): number {
  const parsed = Number.parseInt(process.env.TTS_PCM_SAMPLE_RATE || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_PCM_SAMPLE_RATE;
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

function pcmToMp3(pcm: Buffer, sampleRate: number): Buffer {
  const encoder = new Mp3Encoder(1, sampleRate, 128);
  const blockSizeSamples = 1152;
  const bytesPerSample = 2;
  const chunks: Buffer[] = [];

  for (let offset = 0; offset < pcm.length; offset += blockSizeSamples * bytesPerSample) {
    const chunk = pcm.subarray(offset, offset + blockSizeSamples * bytesPerSample);
    const sampleCount = Math.floor(chunk.byteLength / bytesPerSample);
    if (sampleCount === 0) break;
    const samples = new Int16Array(
      chunk.buffer,
      chunk.byteOffset,
      sampleCount
    );
    const encoded = encoder.encodeBuffer(samples);
    if (encoded.length > 0) chunks.push(Buffer.from(encoded));
  }

  const flush = encoder.flush();
  if (flush.length > 0) chunks.push(Buffer.from(flush));

  return Buffer.concat(chunks);
}

export async function generateTTSAudio(
  text: string,
  voice: string,
  apiKey?: string
): Promise<Buffer> {
  const effectiveApiKey = apiKey || process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL;
  const responseFormat = getTTSResponseFormat();

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
      response_format: responseFormat,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "Unknown error");
    throw new Error(
      `TTS request failed (${response.status}): ${errorText}`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  let audioBuffer: Buffer = Buffer.from(arrayBuffer);

  if (responseFormat === "pcm") {
    audioBuffer = pcmToMp3(audioBuffer, getPCMSampleRate());
  }

  return audioBuffer;
}
