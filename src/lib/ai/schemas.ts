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
  ideas: z
    .string()
    .min(1)
    .describe(
      "Raw HTML string (markdown is FORBIDDEN). Structure: <strong> for key topics, <em> for emphasis, <br><br> between major points, <br> after every bullet/numbered item, &nbsp;&nbsp;&nbsp;&nbsp; for indentation of sub-points. Plain newlines do NOT create line breaks when rendered - always use <br> tags."
    ),
  language: z
    .string()
    .min(1)
    .describe(
      'Raw HTML string containing exactly one <table> built with <table>, <tr>, <th>, <td> tags and columns "English | 中文 | Usage Example". Markdown pipe tables (|) are FORBIDDEN.'
    ),
  communication_strategies: z
    .string()
    .min(1)
    .describe(
      "Raw HTML string (markdown is FORBIDDEN). Each strategy MUST be its own block: <strong>Strategy name (中文名稱)</strong> followed by <em>example phrases</em> and a Traditional Chinese explanation. Separate strategies with <br><br> and separate examples with <br>. Never use markdown asterisks (** or *) and never rely on plain newlines for layout - always use <br> tags."
    ),
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
