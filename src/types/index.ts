export const SPEAKER_VOICE_MAPPINGS: Record<string, string> = {
  "Candidate A": "nova",
  "Candidate B": "alloy",
  "Candidate C": "fable",
  "Candidate D": "echo",
};

export const SPEAKER_COLORS: Record<string, string> = {
  "Candidate A": "bg-blue-50 dark:bg-blue-950 border-l-blue-400",
  "Candidate B": "bg-yellow-50 dark:bg-yellow-950 border-l-yellow-400",
  "Candidate C": "bg-green-50 dark:bg-green-950 border-l-green-400",
  "Candidate D": "bg-pink-50 dark:bg-pink-950 border-l-pink-400",
};

export const SPEAKER_COLORS_HEX: Record<string, string> = {
  "Candidate A": "#E3F2FD",
  "Candidate B": "#FFFDE7",
  "Candidate C": "#E8F5E8",
  "Candidate D": "#FDECEA",
};

export type Speaker = "Candidate A" | "Candidate B" | "Candidate C" | "Candidate D";

export type DialogueMode = "Normal" | "Deeper";
export type InputMethod = "Upload Files" | "Enter Topic";

export interface DialogueItem {
  text: string;
  speaker: Speaker;
}

export interface LearningNotes {
  ideas: string;
  language: string;
  communication_strategies: string;
}

export interface Dialogue {
  scratchpad: string;
  dialogue: DialogueItem[];
  learning_notes: LearningNotes;
}

export interface DiscussionSession {
  id: string;
  userId: string;
  title: string;
  dialogueMode: DialogueMode;
  inputMethod: InputMethod;
  inputText: string | null;
  transcript: DialogueItem[];
  learningNotes: LearningNotes;
  audioUrl: string | null;
  accessCode: string | null;
  charactersCount: number;
  ttsCostHKD: number;
  usedOwnApiKey: boolean;
  createdAt: Date;
}

export interface GenerateRequest {
  inputMethod: InputMethod;
  dialogueMode: DialogueMode;
  text: string;
  apiKey?: string;
}

export interface GenerateResponse {
  dialogue: DialogueItem[];
  learningNotes: LearningNotes;
  scratchpad: string;
  charactersCount: number;
  ttsCostHKD: number;
  title: string;
  extractedText: string;
  generationId: string | null;
  usedOwnApiKey: boolean;
  creditsConsumed: number;
}

export interface TTSRequest {
  text: string;
  voice: string;
  apiKey?: string;
}
