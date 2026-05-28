import { generateObject } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { dialogueSchema, individualResponseSchema, questionExtractionSchema, brainstormSchema, dialogueOnlySchema, dialogueLearningNotesSchema, ideasNotesSchema, languageNotesSchema, strategiesNotesSchema } from "./schemas";
import { buildDialoguePrompt, buildIndividualResponsePrompt, QUESTION_EXTRACTION_SYSTEM, buildQuestionExtractionPrompt, buildBrainstormPrompt, buildDialogueFromBrainstormPrompt, buildDialogueLearningNotesPrompt, buildIdeasNotesPrompt, buildLanguageNotesPrompt, buildStrategiesNotesPrompt } from "./prompts";
import type { Dialogue, DialogueMode } from "@/types";

function getOpenAIClient(apiKey?: string) {
  return createOpenAI({
    apiKey: apiKey || process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });
}

function getModelId(mode: DialogueMode) {
  return mode === "Deeper"
    ? process.env.OPENAI_MODEL_DEEP || "gpt-4.1"
    : process.env.OPENAI_MODEL_NORMAL || "gpt-4.1-mini";
}

function isReasoningModel(modelId: string) {
  return /o[1-4]|gpt-5/i.test(modelId);
}

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
    model: openai(modelId),
    schema: dialogueSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxOutputTokens: isReasoning ? 16000 : 8000,
    maxRetries: 2,
  });

  return object as Dialogue;
}

export async function generateBrainstorm(
  text: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = process.env.OPENAI_MODEL_BRAINSTORM || getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildBrainstormPrompt(text);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: brainstormSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxOutputTokens: 4000,
    maxRetries: 2,
  });

  return object;
}

export async function generateDialogueFromBrainstorm(
  brainstorm: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildDialogueFromBrainstormPrompt(brainstorm);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: dialogueOnlySchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxOutputTokens: isReasoning ? 16000 : 8000,
    maxRetries: 2,
  });

  return object;
}

export async function generateDialogueLearningNotes(
  dialogueText: string,
  brainstorm: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildDialogueLearningNotesPrompt(dialogueText, brainstorm);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: dialogueLearningNotesSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.3 }),
    maxOutputTokens: isReasoning ? 8000 : 4000,
    maxRetries: 2,
  });

  return object;
}

export async function generateIdeasNotes(
  dialogueText: string,
  brainstorm: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildIdeasNotesPrompt(dialogueText, brainstorm);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: ideasNotesSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.3 }),
    maxOutputTokens: 8000,
    maxRetries: 2,
  });

  return object;
}

export async function generateLanguageNotes(
  dialogueText: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildLanguageNotesPrompt(dialogueText);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: languageNotesSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.3 }),
    maxOutputTokens: 8000,
    maxRetries: 2,
  });

  return object;
}

export async function generateStrategiesNotes(
  dialogueText: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildStrategiesNotesPrompt(dialogueText);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: strategiesNotesSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.3 }),
    maxOutputTokens: 8000,
    maxRetries: 2,
  });

  return object;
}

export async function generateIndividualResponse(
  text: string,
  mode: DialogueMode,
  apiKey?: string
) {
  const openai = getOpenAIClient(apiKey);
  const modelId = getModelId(mode);
  const isReasoning = isReasoningModel(modelId);

  const { system, user } = buildIndividualResponsePrompt(text);

  const { object } = await generateObject({
    model: openai(modelId),
    schema: individualResponseSchema,
    system,
    prompt: user,
    ...(isReasoning ? {} : { temperature: 0.5 }),
    maxOutputTokens: isReasoning ? 8000 : 4000,
    maxRetries: 2,
  });

  return object;
}

export async function extractQuestions(
  text: string,
  apiKey?: string
): Promise<string[]> {
  const openai = getOpenAIClient(apiKey);
  const modelId = process.env.OPENAI_MODEL_QUESTION_EXTRACTION || process.env.OPENAI_MODEL_NORMAL || "gpt-4.1-mini";

  const { object } = await generateObject({
    model: openai(modelId),
    schema: questionExtractionSchema,
    system: QUESTION_EXTRACTION_SYSTEM,
    prompt: buildQuestionExtractionPrompt(text),
    temperature: 0,
    maxOutputTokens: 2000,
    maxRetries: 2,
  });

  return object.questions;
}
