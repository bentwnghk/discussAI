import fs from "fs/promises";
import * as pdfParse from "pdf-parse";

export async function extractTextFromPDF(filePath: string): Promise<string> {
  const dataBuffer = await fs.readFile(filePath);
  const data = await (pdfParse as unknown as (data: Buffer) => Promise<{ text: string }>)(dataBuffer);
  return data.text.trim();
}
