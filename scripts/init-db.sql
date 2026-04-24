-- DiscussAI Database Initialization
-- Run: docker exec -i discussai-db psql -U discussai -d discussai < scripts/init-db.sql

BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users (NextAuth expects table name "user")
CREATE TABLE IF NOT EXISTS "user" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "name" TEXT,
  "email" TEXT NOT NULL UNIQUE,
  "emailVerified" TIMESTAMP,
  "image" TEXT,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Accounts (NextAuth expects table name "account")
CREATE TABLE IF NOT EXISTS "account" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" UUID NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "type" VARCHAR(255) NOT NULL,
  "provider" VARCHAR(255) NOT NULL,
  "providerAccountId" VARCHAR(255) NOT NULL,
  "refresh_token" TEXT,
  "access_token" TEXT,
  "expires_at" INTEGER,
  "token_type" VARCHAR(255),
  "scope" TEXT,
  "id_token" TEXT,
  "session_state" TEXT
);

-- Sessions (NextAuth expects table name "session", PK is sessionToken)
CREATE TABLE IF NOT EXISTS "session" (
  "sessionToken" VARCHAR(255) PRIMARY KEY,
  "userId" UUID NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "expires" TIMESTAMP NOT NULL
);

-- Verification Tokens (NextAuth expects table name "verification_token")
CREATE TABLE IF NOT EXISTS "verification_token" (
  "identifier" VARCHAR(255) NOT NULL,
  "token" VARCHAR(255) NOT NULL,
  "expires" TIMESTAMP NOT NULL,
  PRIMARY KEY ("identifier", "token")
);

-- Discussion Sessions (app-specific)
CREATE TABLE IF NOT EXISTS "discussion_sessions" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" UUID NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "title" VARCHAR(500) NOT NULL,
  "dialogueMode" VARCHAR(20) NOT NULL,
  "inputMethod" VARCHAR(20) NOT NULL,
  "inputText" TEXT,
  "transcript" JSONB NOT NULL,
  "learningNotes" JSONB NOT NULL,
  "audioUrl" TEXT,
  "charactersCount" INTEGER NOT NULL DEFAULT 0,
  "ttsCostHKD" REAL NOT NULL DEFAULT 0,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS "idx_account_userId" ON "account"("userId");
CREATE INDEX IF NOT EXISTS "idx_account_provider" ON "account"("provider", "providerAccountId");
CREATE INDEX IF NOT EXISTS "idx_session_userId" ON "session"("userId");
CREATE INDEX IF NOT EXISTS "idx_discussion_sessions_userId" ON "discussion_sessions"("userId");
CREATE INDEX IF NOT EXISTS "idx_discussion_sessions_createdAt" ON "discussion_sessions"("createdAt" DESC);

COMMIT;
