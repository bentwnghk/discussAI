import { NextRequest, NextResponse } from "next/server";
import { readdir, stat, unlink } from "fs/promises";
import path from "path";
import { db } from "@/lib/db";
import { discussionSessions } from "@/lib/db/schema";
import { isNotNull, sql } from "drizzle-orm";
import { AUDIO_TTL_MS } from "@/lib/audio-ttl";

const AUDIO_DIR = path.join(process.cwd(), "tmp", "audio");

export async function POST(req: NextRequest) {
  const cronSecret = process.env.CRON_SECRET;
  if (cronSecret) {
    const authHeader = req.headers.get("authorization");
    if (authHeader !== `Bearer ${cronSecret}`) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  try {
    let deletedFiles = 0;

    const files = await readdir(AUDIO_DIR).catch(() => [] as string[]);
    const expiredFilenames = new Set<string>();

    for (const file of files) {
      const filePath = path.join(AUDIO_DIR, file);
      const fileStat = await stat(filePath);
      if (Date.now() - fileStat.mtimeMs > AUDIO_TTL_MS) {
        await unlink(filePath).catch(() => {});
        expiredFilenames.add(file);
        deletedFiles++;
      }
    }

    if (expiredFilenames.size > 0) {
      const sessionsWithAudio = await db
        .select({
          id: discussionSessions.id,
          audioUrl: discussionSessions.audioUrl,
        })
        .from(discussionSessions)
        .where(isNotNull(discussionSessions.audioUrl));

      for (const session of sessionsWithAudio) {
        if (!session.audioUrl) continue;
        const filename = session.audioUrl.split("/").pop();
        if (filename && expiredFilenames.has(filename)) {
          await db
            .update(discussionSessions)
            .set({ audioUrl: null })
            .where(sql`${discussionSessions.id} = ${session.id}`);
        }
      }
    }

    return NextResponse.json({
      success: true,
      deletedFiles,
      expiredCount: expiredFilenames.size,
    });
  } catch (error) {
    console.error("Audio cleanup error:", error);
    return NextResponse.json(
      { error: "Cleanup failed." },
      { status: 500 }
    );
  }
}
