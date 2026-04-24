import { createOpenAI } from "@ai-sdk/openai";
import { generateText } from "ai";
import fs from "fs/promises";
import path from "path";

export async function extractTextFromImage(
  filePath: string,
  apiKey?: string
): Promise<string> {
  const openai = createOpenAI({
    apiKey: apiKey || process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });

  const data = await fs.readFile(filePath);
  const base64 = data.toString("base64");
  const ext = path.extname(filePath).toLowerCase();
  const mimeType =
    ext === ".jpg" || ext === ".jpeg"
      ? "image/jpeg"
      : ext === ".png"
        ? "image/png"
        : "image/png";

  const { text } = await generateText({
    model: openai("gpt-4.1-mini"),
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            image: `data:${mimeType};base64,${base64}`,
          },
          {
            type: "text",
            text: "Extract all the computer-readable text from this image as accurately as possible. Avoid commentary, return only the extracted text.",
          },
        ],
      },
    ],
    maxOutputTokens: 32768,
    temperature: 0,
    maxRetries: 2,
  });

  return text.trim();
}
