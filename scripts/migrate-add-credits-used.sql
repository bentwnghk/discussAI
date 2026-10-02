-- Store the actual credits consumed per session so history display
-- does not change when GENERATION_COST / RESPONSE_COST env vars change.
ALTER TABLE discussion_sessions
ADD COLUMN IF NOT EXISTS "creditsUsed" INTEGER;

-- Sessions generated with the user's own API key consumed no credits.
UPDATE discussion_sessions
SET "creditsUsed" = 0
WHERE "usedOwnApiKey" = true
  AND "creditsUsed" IS NULL;

-- Backfill from credit_transactions (same source as the Usage History
-- table on /credits): match each session to the nearest preceding
-- 'generation' transaction of the matching type for the same user.
UPDATE discussion_sessions ds
SET "creditsUsed" = sub.amount
FROM (
  SELECT DISTINCT ON (s.id) s.id, ABS(ct.amount) AS amount
  FROM discussion_sessions s
  JOIN LATERAL (
    SELECT ct.amount
    FROM credit_transactions ct
    WHERE ct."userId" = s."userId"
      AND ct.type = 'generation'
      AND ct."createdAt" <= s."createdAt"
      AND ct.description = CASE WHEN s."sessionType" = 'response'
                                THEN 'Individual response generation'
                                ELSE 'Discussion generation' END
    ORDER BY ct."createdAt" DESC
    LIMIT 1
  ) ct ON true
  WHERE s."usedOwnApiKey" = false
    AND s."creditsUsed" IS NULL
) sub
WHERE ds.id = sub.id;
