import fs from "fs/promises";

export async function extractTextFromPDF(filePath: string): Promise<string> {
  const g = globalThis as Record<string, unknown>;
  if (typeof g.DOMMatrix === "undefined") g.DOMMatrix = class {};
  if (typeof g.ImageData === "undefined") g.ImageData = class {};
  if (typeof g.Path2D === "undefined") g.Path2D = class {};

  const mod = await import("pdf-parse");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pdfParse = (mod as any).default || mod;
  const dataBuffer = await fs.readFile(filePath);
  const data: { text: string } = await pdfParse(dataBuffer);
  return data.text.trim();
}
