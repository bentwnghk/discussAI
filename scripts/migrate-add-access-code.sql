ALTER TABLE discussion_sessions ADD COLUMN IF NOT EXISTS "accessCode" VARCHAR(8) UNIQUE;

DO $$
DECLARE
    rec RECORD;
    code TEXT;
BEGIN
    FOR rec IN SELECT id FROM discussion_sessions WHERE "accessCode" IS NULL LOOP
        LOOP
            code := upper(substring(md5(random()::text || clock_timestamp()::text) from 1 for 6));
            EXIT WHEN NOT EXISTS (SELECT 1 FROM discussion_sessions WHERE "accessCode" = code);
        END LOOP;
        UPDATE discussion_sessions SET "accessCode" = code WHERE id = rec.id;
        RAISE NOTICE '  % -> %', rec.id, code;
    END LOOP;
END;
$$;
