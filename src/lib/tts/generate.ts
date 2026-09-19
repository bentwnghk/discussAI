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

function looksLikeMp3(buf: Buffer): boolean {
  if (buf.length >= 3 && buf[0] === 0x49 && buf[1] === 0x44 && buf[2] === 0x33) {
    return true;
  }
  return buf.length >= 2 && buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0;
}

function extractErrorMessage(text: string): string {
  let message = text.slice(0, 500);
  try {
    const parsed = JSON.parse(text) as {
      error?: { message?: string } | string;
      message?: string;
    };
    message =
      (typeof parsed.error === "string"
        ? parsed.error
        : parsed.error?.message) ||
      parsed.message ||
      message;
  } catch {
    // keep raw text
  }
  return message;
}

function decodeJsonAudioResponse(
  body: string
): { audio: Buffer; mimeType: string } | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;

  type Source = { data: unknown; mime: unknown };
  const sources: Source[] = [];

  if (Array.isArray(obj.candidates) && obj.candidates.length > 0) {
    const first = obj.candidates[0] as Record<string, unknown> | null;
    const content = first?.content as Record<string, unknown> | undefined;
    if (Array.isArray(content?.parts)) {
      for (const part of content.parts as Record<string, unknown>[]) {
        const inline = part?.inlineData as Record<string, unknown> | undefined;
        if (inline) sources.push({ data: inline.data, mime: inline.mimeType });
      }
    }
  }

  sources.push({
    data: obj.data,
    mime: obj.mime_type ?? obj.mimeType,
  });
  sources.push({ data: obj.b64_json, mime: obj.mime_type });

  if (typeof obj.audio === "string") {
    sources.push({ data: obj.audio, mime: obj.mime_type });
  } else if (obj.audio && typeof obj.audio === "object") {
    const audioObj = obj.audio as Record<string, unknown>;
    sources.push({ data: audioObj.data, mime: audioObj.mime_type ?? audioObj.mimeType });
  }

  for (const src of sources) {
    if (typeof src.data !== "string" || src.data.length === 0) continue;
    const audio = Buffer.from(src.data, "base64");
    if (audio.length === 0) continue;
    return {
      audio,
      mimeType: typeof src.mime === "string" ? src.mime : "",
    };
  }
  return null;
}

function sampleRateFromMime(mime: string): number | null {
  const match = /rate=(\d{4,6})/i.exec(mime);
  if (!match) return null;
  const rate = Number.parseInt(match[1], 10);
  return Number.isFinite(rate) && rate > 0 ? rate : null;
}

function finalizeJsonAudio(audio: Buffer, mimeType: string): Buffer {
  const mime = mimeType.toLowerCase();

  if (mime.includes("mpeg") || mime.includes("mp3")) return audio;
  if (looksLikeMp3(audio)) return audio;
  if (mime.includes("wav") || (audio.length >= 4 && audio.toString("ascii", 0, 4) === "RIFF")) {
    throw new Error(
      `TTS provider returned WAV audio, which is not supported. Configure the provider to return mp3 or pcm.`
    );
  }

  const rate = sampleRateFromMime(mime) ?? getPCMSampleRate();
  console.warn(
    `TTS: decoded base64 audio from JSON response (${mimeType || "unknown mime"}); transcoding pcm@${rate}Hz to mp3.`
  );
  return pcmToMp3(audio, rate);
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

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("json") || contentType.startsWith("text/")) {
    const bodyText = await response.text().catch(() => "");
    const decoded = decodeJsonAudioResponse(bodyText);
    if (decoded) {
      return finalizeJsonAudio(decoded.audio, decoded.mimeType);
    }
    throw new Error(
      `TTS provider returned a non-audio response (${contentType}): ${extractErrorMessage(bodyText)}`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  let audioBuffer: Buffer = Buffer.from(arrayBuffer);

  if (audioBuffer.length === 0) {
    throw new Error("TTS provider returned an empty audio response.");
  }

  if (responseFormat === "pcm") {
    if (looksLikeMp3(audioBuffer)) {
      console.warn(
        `TTS: requested response_format=pcm but provider returned mp3; using mp3 as-is.`
      );
    } else {
      audioBuffer = pcmToMp3(audioBuffer, getPCMSampleRate());
    }
  }

  return audioBuffer;
}
