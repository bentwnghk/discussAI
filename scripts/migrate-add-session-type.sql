ALTER TABLE discussion_sessions
ADD COLUMN IF NOT EXISTS "sessionType" varchar(20) NOT NULL DEFAULT 'discussion';
