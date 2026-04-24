-- DiscussAI Database Initialization
-- Run: docker exec -i discussai-db psql -U discussai -d discussai < scripts/init-db.sql

BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users
CREATE TABLE IF NOT EXISTS "users" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "name" TEXT,
  "email" TEXT NOT NULL UNIQUE,
  "email_verified" TIMESTAMP,
  "image" TEXT,
  "created_at" TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Accounts (NextAuth OAuth)
CREATE TABLE IF NOT EXISTS "accounts" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "user_id" UUID NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "type" VARCHAR(255) NOT NULL,
  "provider" VARCHAR(255) NOT NULL,
  "provider_account_id" VARCHAR(255) NOT NULL,
  "refresh_token" TEXT,
  "access_token" TEXT,
  "expires_at" INTEGER,
  "token_type" VARCHAR(255),
  "scope" TEXT,
  "id_token" TEXT,
  "session_state" TEXT
);

-- Sessions (NextAuth)
CREATE TABLE IF NOT EXISTS "sessions" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "session_token" VARCHAR(255) NOT NULL UNIQUE,
  "user_id" UUID NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "expires" TIMESTAMP NOT NULL
);

-- Verification Tokens (NextAuth)
CREATE TABLE IF NOT EXISTS "verification_tokens" (
  "identifier" VARCHAR(255) NOT NULL,
  "token" VARCHAR(255) NOT NULL UNIQUE,
  "expires" TIMESTAMP NOT NULL
);

-- Discussion Sessions
CREATE TABLE IF NOT EXISTS "discussion_sessions" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "user_id" UUID NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "title" VARCHAR(500) NOT NULL,
  "dialogue_mode" VARCHAR(20) NOT NULL,
  "input_method" VARCHAR(20) NOT NULL,
  "input_text" TEXT,
  "transcript" JSONB NOT NULL,
  "learning_notes" JSONB NOT NULL,
  "audio_url" TEXT,
  "characters_count" INTEGER NOT NULL DEFAULT 0,
  "tts_cost_hkd" REAL NOT NULL DEFAULT 0,
  "created_at" TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS "idx_accounts_user_id" ON "accounts"("user_id");
CREATE INDEX IF NOT EXISTS "idx_accounts_provider" ON "accounts"("provider", "provider_account_id");
CREATE INDEX IF NOT EXISTS "idx_sessions_user_id" ON "sessions"("user_id");
CREATE INDEX IF NOT EXISTS "idx_sessions_token" ON "sessions"("session_token");
CREATE INDEX IF NOT EXISTS "idx_verification_tokens_token" ON "verification_tokens"("token");
CREATE INDEX IF NOT EXISTS "idx_discussion_sessions_user_id" ON "discussion_sessions"("user_id");
CREATE INDEX IF NOT EXISTS "idx_discussion_sessions_created_at" ON "discussion_sessions"("created_at" DESC);

COMMIT;
