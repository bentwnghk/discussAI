import { generateObject } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { dialogueSchema, individualResponseSchema, questionExtractionSchema } from "./schemas";
import { buildDialoguePrompt, buildIndividualResponsePrompt, QUESTION_EXTRACTION_SYSTEM, buildQuestionExtractionPrompt } from "./prompts";
import { normalizeLearningNotes } from "./notes-formatter";
import type { Dialogue, DialogueMode } from "@/types";

function getOpenAIClient(apiKey?: string) {
  return createOpenAI({
    apiKey: apiKey || process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
    fetch: getProviderFetch(),
  });
}

function getModelId(mode: DialogueMode) {
  return mode === "Deeper"
    ? process.env.OPENAI_MODEL_DEEP || "gpt-4.1"
    : process.env.OPENAI_MODEL_NORMAL || "gpt-4.1-mini";
}

function isReasoningModel(modelId: string) {
  return /o[1-4]|gpt-5|glm|deepseek/i.test(modelId);
}

const RESPONSE_FORMAT_MODES = new Set(["json_schema", "json_object", "none", "auto"]);

const STRUCTURED_OUTPUT_MODEL_PATTERN = /^(gpt-4\.1|gpt-4o|gpt-[5-9]|o[1-9](?:-mini)?|chatgpt|glm)/i;

function resolveTargetFormat(mode: string, model?: string): string {
  if (mode !== "auto") return mode;
  return model && STRUCTURED_OUTPUT_MODEL_PATTERN.test(model) ? "json_schema" : "json_object";
}

function getProviderFetch() {
  const mode = (process.env.OPENAI_RESPONSE_FORMAT ?? "json_schema").toLowerCase();
  if (!RESPONSE_FORMAT_MODES.has(mode) || mode === "json_schema") return undefined;

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    if (init?.body && typeof init.body === "string") {
      try {
        const body = JSON.parse(init.body) as { model?: string; response_format?: { type?: string } };
        if (body.response_format?.type === "json_schema") {
          const target = resolveTargetFormat(mode, body.model);
          if (target === "json_object") {
            body.response_format = { type: "json_object" };
          } else if (target === "none") {
            delete body.response_format;
          }
          if (target !== "json_schema") {
            init = { ...init, body: JSON.stringify(body) };
          }
        }
      } catch {
        // non-JSON body; send unchanged
      }
    }
    return fetch(input, init);
  };
}

function getReasoningProviderOptions() {
  const effort = process.env.OPENAI_REASONING_EFFORT;
  if (!effort) return undefined;
  return { openai: { reasoningEffort: effort as "none" | "low" | "medium" | "high" } };
}

function extractJsonFromText(text: string): string | null {
  const fences = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)].map((m) => m[1]);
  let candidate =
    fences.length > 0
      ? fences.sort((a, b) => b.length - a.length)[0].trim()
      : text;

  const start = candidate.search(/[{[]/);
  if (start === -1) return null;
  candidate = candidate.slice(start);

  const end = Math.max(candidate.lastIndexOf("}"), candidate.lastIndexOf("]"));
  if (end !== -1) candidate = candidate.slice(0, end + 1);

  return candidate;
}

const repairModelText = async ({ text }: { text: string }): Promise<string | null> =>
  extractJsonFromText(text);

export async function generateDialogue(
  text: string,
  mode: DialogueMode,
  apiKey?: string
): Promise<Dialogue> {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildDialoguePrompt(text);

  const { object } = await generateObject({
    model: openai.chat(modelId),
    schema: dialogueSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxOutputTokens: isReasoning ? 16000 : 8000,
    maxRetries: 2,
    providerOptions: getReasoningProviderOptions(),
    experimental_repairText: repairModelText,
  });

  const result = object as Dialogue;
  return { ...result, learning_notes: normalizeLearningNotes(result.learning_notes) };
}

export async function generateIndividualResponse(
  text: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildIndividualResponsePrompt(text, mode);

  const { object } = await generateObject({
    model: openai.chat(modelId),
    schema: individualResponseSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxOutputTokens: isReasoning ? 8000 : 4000,
    maxRetries: 2,
    providerOptions: getReasoningProviderOptions(),
    experimental_repairText: repairModelText,
  });

  return { ...object, learning_notes: normalizeLearningNotes(object.learning_notes) };
}

export async function extractQuestions(
  text: string,
  apiKey?: string
): Promise<string[]> {
  const openai = getOpenAIClient(apiKey);
  const modelId = process.env.OPENAI_MODEL_QUESTION_EXTRACTION || process.env.OPENAI_MODEL_NORMAL || "gpt-4.1-mini";

  const { object } = await generateObject({
    model: openai.chat(modelId),
    schema: questionExtractionSchema,
    system: QUESTION_EXTRACTION_SYSTEM,
    prompt: buildQuestionExtractionPrompt(text),
    temperature: 0,
    maxOutputTokens: 2000,
    maxRetries: 2,
    providerOptions: getReasoningProviderOptions(),
    experimental_repairText: repairModelText,
  });

  return object.questions;
}
