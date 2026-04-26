-- Add usedOwnApiKey column to discussion_sessions table
ALTER TABLE "discussion_sessions" ADD COLUMN IF NOT EXISTS "usedOwnApiKey" BOOLEAN NOT NULL DEFAULT false;
