import {
  pgTable,
  text,
  timestamp,
  uuid,
  varchar,
  integer,
  real,
  jsonb,
  primaryKey,
  boolean,
} from "drizzle-orm/pg-core";

export const users = pgTable("user", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name"),
  email: text("email").notNull().unique(),
  emailVerified: timestamp("emailVerified", { mode: "date" }),
  image: text("image"),
  apiKey: text("apiKey"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});

export const accounts = pgTable("account", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  type: varchar("type", { length: 255 }).notNull(),
  provider: varchar("provider", { length: 255 }).notNull(),
  providerAccountId: varchar("providerAccountId", { length: 255 }).notNull(),
  refresh_token: text("refresh_token"),
  access_token: text("access_token"),
  expires_at: integer("expires_at"),
  token_type: varchar("token_type", { length: 255 }),
  scope: text("scope"),
  id_token: text("id_token"),
  session_state: text("session_state"),
});

export const sessions = pgTable("session", {
  sessionToken: varchar("sessionToken", { length: 255 }).primaryKey(),
  userId: uuid("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expires: timestamp("expires", { mode: "date" }).notNull(),
});

export const verificationTokens = pgTable(
  "verification_token",
  {
    identifier: varchar("identifier", { length: 255 }).notNull(),
    token: varchar("token", { length: 255 }).notNull(),
    expires: timestamp("expires", { mode: "date" }).notNull(),
  },
  (t) => [primaryKey({ columns: [t.identifier, t.token] })]
);

export const discussionSessions = pgTable("discussion_sessions", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  title: varchar("title", { length: 500 }).notNull(),
  dialogueMode: varchar("dialogueMode", { length: 20 }).notNull(),
  inputMethod: varchar("inputMethod", { length: 20 }).notNull(),
  inputText: text("inputText"),
  transcript: jsonb("transcript").notNull(),
  learningNotes: jsonb("learningNotes").notNull(),
  audioUrl: text("audioUrl"),
  accessCode: varchar("accessCode", { length: 8 }).unique(),
  charactersCount: integer("charactersCount").notNull().default(0),
  ttsCostHKD: real("ttsCostHKD").notNull().default(0),
  usedOwnApiKey: boolean("usedOwnApiKey").notNull().default(false),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});

export const credits = pgTable("credits", {
  userId: uuid("userId")
    .primaryKey()
    .references(() => users.id, { onDelete: "cascade" }),
  balance: integer("balance").notNull().default(0),
  updatedAt: timestamp("updatedAt", { mode: "date" }).defaultNow().notNull(),
});

export const creditTransactions = pgTable("credit_transactions", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  amount: integer("amount").notNull(),
  type: varchar("type", { length: 30 }).notNull(),
  description: text("description"),
  stripeSessionId: text("stripeSessionId"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});

export const purchases = pgTable("purchases", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  stripeSessionId: text("stripeSessionId").notNull().unique(),
  stripePaymentIntentId: text("stripePaymentIntentId"),
  planName: varchar("planName", { length: 20 }).notNull(),
  creditsAmount: integer("creditsAmount").notNull(),
  amountHKD: real("amountHKD").notNull(),
  status: varchar("status", { length: 20 }).notNull().default("pending"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});
