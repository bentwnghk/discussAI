-- Add apiKey column to users table for existing databases
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS "apiKey" TEXT;
