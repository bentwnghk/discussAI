import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { dialogueSchema } from "./schemas";
import { buildDialoguePrompt } from "./prompts";
import type { Dialogue, DialogueMode } from "@/types";

function getOpenAIClient(apiKey?: string) {
  return createOpenAI({
    apiKey: apiKey || process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });
}

export async function generateDialogue(
  text: string,
  mode: DialogueMode,
  apiKey?: string
): Promise<Dialogue> {
  const openai = getOpenAIClient(apiKey);
  const modelId =
    mode === "Deeper"
      ? process.env.OPENAI_MODEL_DEEP || "gpt-4.1"
      : process.env.OPENAI_MODEL_NORMAL || "gpt-4.1-mini";

  const isReasoning = /o[1-4]|gpt-5/i.test(modelId);

  const { system, user } = buildDialoguePrompt(text);

  const { text: raw } = await generateText({
    model: openai.chat(modelId),
    system,
    prompt: user + "\n\nReturn ONLY valid JSON matching the schema. No markdown fences.",
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxRetries: 2,
  });

  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  const parsed = JSON.parse(cleaned);
  return dialogueSchema.parse(parsed) as Dialogue;
}
