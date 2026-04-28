import "dotenv/config";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { isNull, eq } from "drizzle-orm";
import { generateUniqueAccessCode } from "@/lib/db/access-code";

async function main() {
  const sessions = await db
    .select({ id: discussionSessions.id })
    .from(discussionSessions)
    .where(isNull(discussionSessions.accessCode));

  if (sessions.length === 0) {
    console.log("No sessions need access codes.");
    return;
  }

  console.log(`Generating access codes for ${sessions.length} sessions...`);

  for (let i = 0; i < sessions.length; i++) {
    const code = await generateUniqueAccessCode();
    await db
      .update(discussionSessions)
      .set({ accessCode: code })
      .where(eq(discussionSessions.id, sessions[i].id));
    console.log(`  [${i + 1}/${sessions.length}] ${sessions[i].id} → ${code}`);
  }

  console.log("Done.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
