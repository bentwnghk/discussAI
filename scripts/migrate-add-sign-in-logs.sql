-- Add sign-in logs table for admin dashboard
CREATE TABLE IF NOT EXISTS "sign_in_logs" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" UUID NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "provider" VARCHAR(50) NOT NULL,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS "idx_sign_in_logs_userId" ON "sign_in_logs"("userId");
CREATE INDEX IF NOT EXISTS "idx_sign_in_logs_createdAt" ON "sign_in_logs"("createdAt" DESC);
