import fs from "fs/promises";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText } from "ai";

export async function extractTextFromPDF(
  filePath: string,
  apiKey?: string
): Promise<string> {
  const g = globalThis as Record<string, unknown>;
  if (typeof g.DOMMatrix === "undefined") g.DOMMatrix = class {};
  if (typeof g.ImageData === "undefined") g.ImageData = class {};
  if (typeof g.Path2D === "undefined") g.Path2D = class {};

  const mod = await import("pdf-parse");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pdfParse = (mod as any).default || mod;
  const dataBuffer = await fs.readFile(filePath);
  const data: { text: string } = await pdfParse(dataBuffer);
  const text = data.text.trim();

  if (text.length > 0) return text;

  return ocrPdfPages(filePath, apiKey);
}

async function ocrPdfPages(
  filePath: string,
  apiKey?: string
): Promise<string> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pdfjsLib = (await import("pdfjs-dist/legacy/build/pdf.mjs")) as any;
  const { createCanvas } = await import("canvas");
  const fileData = new Uint8Array(await fs.readFile(filePath));
  const pdf = await pdfjsLib.getDocument({ data: fileData }).promise;
  const pageTexts: string[] = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 2.0 });
    const canvas = createCanvas(viewport.width, viewport.height);
    const ctx = canvas.getContext("2d");

    await page.render({ canvasContext: ctx, viewport }).promise;

    const pngBuffer = canvas.toBuffer("image/png");
    const base64 = pngBuffer.toString("base64");
    const ocrText = await ocrImage(base64, apiKey);
    if (ocrText.trim()) pageTexts.push(ocrText.trim());
  }

  return pageTexts.join("\n\n");
}

async function ocrImage(
  base64: string,
  apiKey?: string
): Promise<string> {
  const openai = createOpenAI({
    apiKey: apiKey || process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });

  const { text } = await generateText({
    model: openai.chat("gpt-5-nano"),
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            image: `data:image/png;base64,${base64}`,
          },
          {
            type: "text",
            text: "Extract all the computer-readable text from this image as accurately as possible. Avoid commentary, return only the extracted text.",
          },
        ],
      },
    ],
    maxOutputTokens: 6000,
    maxRetries: 2,
  });

  return text.trim();
}
