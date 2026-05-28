import { z } from "zod";

const speakerSchema = z.enum([
  "Candidate A",
  "Candidate B",
  "Candidate C",
  "Candidate D",
]);

export const dialogueItemSchema = z.object({
  text: z.string().min(1),
  speaker: speakerSchema,
});

export const learningNotesSchema = z.object({
  ideas: z.string().min(1),
  language: z.string().min(1),
  communication_strategies: z.string().min(1),
});

export const ideasNotesSchema = z.object({
  ideas: z.string().min(1),
});

export const languageNotesSchema = z.object({
  language: z.string().min(1),
});

export const strategiesNotesSchema = z.object({
  communication_strategies: z.string().min(1),
});

export const brainstormSchema = z.object({
  scratchpad: z.string(),
  topic_summary: z.string(),
  question_prompts: z.array(z.string()),
  key_points: z.array(z.string()),
});

export const dialogueOnlySchema = z.object({
  dialogue: z.array(dialogueItemSchema).min(4),
});

export const dialogueLearningNotesSchema = z.object({
  learning_notes: learningNotesSchema,
});

export const dialogueSchema = z.object({
  scratchpad: z.string(),
  dialogue: z.array(dialogueItemSchema).min(4),
  learning_notes: learningNotesSchema,
});

const responseSpeakerSchema = z.enum(["Speaker"]);

export const responseItemSchema = z.object({
  text: z.string().min(1),
  speaker: responseSpeakerSchema,
});

export const individualResponseSchema = z.object({
  scratchpad: z.string(),
  response: z.array(responseItemSchema).min(2),
  learning_notes: learningNotesSchema,
});

export const questionExtractionSchema = z.object({
  questions: z.array(z.string()),
});
