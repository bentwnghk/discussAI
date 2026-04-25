import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";
import { eq } from "drizzle-orm";

export async function getUserApiKey(
  userId: string
): Promise<string | undefined> {
  const [user] = await db
    .select({ apiKey: users.apiKey })
    .from(users)
    .where(eq(users.id, userId));

  return user?.apiKey || undefined;
}
