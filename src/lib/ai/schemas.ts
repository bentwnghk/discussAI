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

export const dialogueSchema = z.object({
  scratchpad: z.string(),
  dialogue: z.array(dialogueItemSchema).min(4),
  learning_notes: learningNotesSchema,
});
