import path from "path";
import { extractTextFromPDF } from "./pdf";
import { extractTextFromDOCX } from "./docx";
import { extractTextFromImage } from "./image-ocr";

const SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".docx", ".pdf"];

function getFileExtension(filename: string): string {
  return path.extname(filename).toLowerCase();
}

export function isSupportedFile(filename: string): boolean {
  return SUPPORTED_EXTENSIONS.includes(getFileExtension(filename));
}

export async function extractTextFromFile(
  filePath: string,
  apiKey?: string
): Promise<string> {
  const ext = getFileExtension(filePath);

  switch (ext) {
    case ".pdf":
      return extractTextFromPDF(filePath, apiKey);
    case ".docx":
      return extractTextFromDOCX(filePath);
    case ".jpg":
    case ".jpeg":
    case ".png":
      return extractTextFromImage(filePath, apiKey);
    default:
      throw new Error(`Unsupported file type: ${ext}`);
  }
}
