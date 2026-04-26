BEGIN;

CREATE TABLE IF NOT EXISTS "credits" (
  "userId" UUID PRIMARY KEY REFERENCES "user"("id") ON DELETE CASCADE,
  "balance" INTEGER NOT NULL DEFAULT 0,
  "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "credit_transactions" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" UUID NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "amount" INTEGER NOT NULL,
  "type" VARCHAR(30) NOT NULL,
  "description" TEXT,
  "stripeSessionId" TEXT,
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "purchases" (
  "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" UUID NOT NULL REFERENCES "user"("id") ON DELETE CASCADE,
  "stripeSessionId" TEXT NOT NULL UNIQUE,
  "stripePaymentIntentId" TEXT,
  "planName" VARCHAR(20) NOT NULL,
  "creditsAmount" INTEGER NOT NULL,
  "amountHKD" REAL NOT NULL,
  "status" VARCHAR(20) NOT NULL DEFAULT 'pending',
  "createdAt" TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS "idx_credit_transactions_userId" ON "credit_transactions"("userId");
CREATE INDEX IF NOT EXISTS "idx_credit_transactions_createdAt" ON "credit_transactions"("createdAt" DESC);
CREATE INDEX IF NOT EXISTS "idx_purchases_userId" ON "purchases"("userId");
CREATE INDEX IF NOT EXISTS "idx_purchases_createdAt" ON "purchases"("createdAt" DESC);

COMMIT;
