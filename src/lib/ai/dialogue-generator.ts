import { generateObject } from "ai";
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

  const { object } = await generateObject({
    model: openai.chat(modelId),
    schema: dialogueSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxRetries: 2,
  });

  return object as Dialogue;
}
