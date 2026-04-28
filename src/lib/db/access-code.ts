import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { customAlphabet } from "nanoid";

const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const generateCode = customAlphabet(ALPHABET, 6);

export async function generateUniqueAccessCode(): Promise<string> {
  for (let i = 0; i < 10; i++) {
    const code = generateCode();
    const existing = await db
      .select({ id: discussionSessions.id })
      .from(discussionSessions)
      .where(eq(discussionSessions.accessCode, code))
      .limit(1);
    if (existing.length === 0) return code;
  }
  throw new Error("Failed to generate unique access code");
}
