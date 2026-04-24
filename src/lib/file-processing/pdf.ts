import fs from "fs/promises";

export async function extractTextFromPDF(filePath: string): Promise<string> {
  const g = globalThis as Record<string, unknown>;
  if (typeof g.DOMMatrix === "undefined") g.DOMMatrix = class {};
  if (typeof g.ImageData === "undefined") g.ImageData = class {};
  if (typeof g.Path2D === "undefined") g.Path2D = class {};

  const mod = await import("pdf-parse");
  const pdfParse = (mod as unknown as { default?: Function; (buf: Buffer): Promise<{ text: string }> }).default
    || (mod as unknown as (buf: Buffer) => Promise<{ text: string }>);
  const dataBuffer = await fs.readFile(filePath);
  const data = await (pdfParse as (data: Buffer) => Promise<{ text: string }>)(
    dataBuffer
  );
  return data.text.trim();
}
