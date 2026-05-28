This document provides essential guidelines and technical references for AI agents (and human developers) working on the **Mr.🆖 DiscussAI** repository. Adhere to these patterns to ensure consistency, security, and maintainability.

---

## Development Workflow & Commands

The project uses **npm** as the primary package manager (Node >= 18).

### Core Commands

- **Install Dependencies**: `npm install`
- **Development Server**: `npm run dev` (Runs at `http://localhost:3000`)
- **Build Project**: `npm run build`
- **Start Production**: `npm run start`
- **Linting**: `npm run lint`

### Database (Drizzle ORM)

- **Generate Migrations**: `npm run db:generate`
- **Push Schema**: `npm run db:push`
- **Run Migrations**: `npm run db:migrate`
- **Drizzle Studio**: `npm run db:studio`

### Testing

- **Status**: Currently, there are no automated tests in the codebase.
- **Guideline**: If adding tests, use **Vitest** following standard Next.js patterns. Place test files next to the code they test (e.g., `ComponentName.test.tsx`) or in a `__tests__` directory.

### Docker

- **Dockerfile**: Multi-stage build on `node:22-alpine`, standalone output, runs as non-root `nextjs` user, exposes port 3000.
- **docker-compose.yml**: Two services — `db` (PostgreSQL 16 Alpine, port 5432, runs `scripts/init-db.sql` on init) and `discussai` (port 3000, depends on healthy db).
- **Build & Run**: `docker compose up --build`

### CI/CD

- **`.github/workflows/docker-image.yml`**: Pushes multi-arch (amd64/arm64) Docker image to Docker Hub on `dev` branch pushes.
- **`.github/workflows/ghcr.yml`**: Pushes Docker image to GitHub Container Registry on `main`/`dev` pushes and `v*` tags.

---

## Project Structure

```
src/
├── middleware.ts                # Auth middleware (route protection, redirects)
├── app/                        # Next.js App Router (Pages, API routes, Layouts)
│   ├── globals.css             # Tailwind v4 CSS-first config, OKLCH theme variables
│   ├── layout.tsx              # Root layout (fonts, providers, header/footer)
│   ├── page.tsx                # Landing page (redirects to /discuss if authenticated)
│   ├── (app)/                  # Authenticated route group (server-side auth guard in layout)
│   │   ├── credits/page.tsx    # Credit purchase page
│   │   ├── discuss/            # Main discussion generation page + history detail
│   │   └── history/            # History listing + individual session view
│   ├── (auth)/                 # Unauthenticated route group
│   │   ├── auth/error/         # Auth error page
│   │   └── login/              # Login page
│   └── api/                    # API route handlers (see Backend section)
├── components/
│   ├── providers.tsx           # Client providers: Session, ApiKey, Credits, Toaster
│   ├── settings-dialog.tsx     # Settings dialog
│   ├── sign-in.tsx             # Sign-in component
│   ├── discuss/                # Core discussion feature components
│   │   ├── audio-player.tsx   # Audio playback for generated dialogue
│   │   ├── file-upload.tsx    # File upload (PDF, DOCX, images)
│   │   ├── learning-notes.tsx # Learning notes display
│   │   └── transcript-display.tsx # Dialogue transcript renderer
│   ├── landing/                # Public landing/marketing page
│   ├── layout/                 # Header, footer
│   └── ui/                     # Shadcn UI primitives (do not modify directly)
├── hooks/
│   ├── use-api-key.tsx         # React Context for user API key management
│   └── use-credits.tsx         # React Context for credit balance tracking
├── lib/
│   ├── auth.ts                 # NextAuth v5 config (Google OAuth, JWT strategy, Drizzle adapter)
│   ├── stripe.ts               # Stripe client + credit plan definitions
│   ├── utils.ts                # cn() utility (clsx + tailwind-merge)
│   ├── pdf-client.ts           # Client-side PDF processing (pdfjs-dist + canvas)
│   ├── ai/
│   │   ├── dialogue-generator.ts # AI dialogue generation (Vercel AI SDK)
│   │   ├── prompts.ts          # System/user prompts for HKDSE discussion simulation
│   │   └── schemas.ts          # Zod schemas for dialogue output validation
│   ├── db/
│   │   ├── index.ts            # Drizzle ORM setup (postgres-js driver, schema export)
│   │   ├── schema.ts           # 8 tables: users, accounts, sessions, verificationTokens, discussionSessions, credits, creditTransactions, purchases
│   │   ├── credits.ts          # Credit system: balance, deduct, refund, purchase tracking
│   │   └── user-api-key.ts     # User API key CRUD
│   ├── export/
│   │   └── docx-generator.ts   # DOCX generation (transcript + learning notes)
│   ├── file-processing/
│   │   ├── index.ts            # File type router (PDF, DOCX, images)
│   │   ├── pdf.ts              # Server-side PDF text extraction (pdf-parse)
│   │   ├── docx.ts             # DOCX text extraction (mammoth)
│   │   └── image-ocr.ts        # Image OCR via OpenAI vision
│   └── tts/
│       └── generate.ts         # TTS via OpenAI-compatible speech API
└── types/
    └── index.ts                # TypeScript types + speaker mappings/colors
scripts/                        # SQL migrations (init-db.sql + incremental migrations)
```

---

## Code Style & Conventions

### 1. TypeScript & Types

- **Strict Mode**: `strict: true` is enabled in `tsconfig.json`. Always provide explicit types for function parameters and return values.
- **Global Types**: Core types (e.g., `DiscussionSession`, `Speaker`, speaker color mappings) are defined in `src/types/index.ts`. Check this file before creating new interfaces.
- **Zod**: Use **Zod v4** for schema validation, especially for AI response parsing (`src/lib/ai/schemas.ts`) and API request bodies.

### 2. React & Next.js

- **App Router**: This project uses the Next.js App Router with route groups `(app)` for authenticated and `(auth)` for unauthenticated pages.
- **Server Components**: Default to Server Components. Only add `"use client"` when browser APIs or React hooks (state, effects) are needed.
- **Route Groups**: `(app)` has a server-side auth guard in its layout. `(auth)` contains login/error pages.

### 3. Components & UI

- **Shadcn UI**: UI primitives are located in `@/components/ui`. Do not modify them directly; extend them or create wrappers.
- **Styling**: Uses **Tailwind CSS v4** with CSS-first configuration (no `tailwind.config.ts`). Theme uses OKLCH color variables. Follow mobile-first responsive design patterns.
- **Dark Mode**: Uses `next-themes` with `darkMode: ["class"]`. Ensure all new UI elements support both light and dark modes using Tailwind `dark:` classes.
- **Icons**: Use **lucide-react**.
- **Fonts**: Poppins (headings) + Plus Jakarta Sans (body) via `next/font/google`.
- **Animations**: Use `motion` (Framer Motion successor) and `tw-animate-css`.

### 4. State Management

- **Server State**: Next.js RSC with `auth()` calls for session data.
- **Client State**: React Context via custom hooks (`useApiKey`, `useCredits`).
- **Provider Hierarchy**: `SessionProvider` > `ApiKeyProvider` > `CreditsProvider`.

### 5. Imports

- **Path Alias**: Always use the `@/` prefix for absolute imports from the `src` directory.
- **Ordering**:
  1. React/Next.js core
  2. Third-party libraries
  3. Components (Internal/UI)
  4. Hooks
  5. Lib & Utils
  6. Types

---

## Authentication (NextAuth v5)

The project uses **next-auth v5 (beta.31)** with **Google OAuth** as the sole provider.

- **`src/lib/auth.ts`**: Full config with `@auth/drizzle-adapter` (PostgreSQL). Session strategy is **JWT** (not database). Includes `ensureCreditsRecord()` callback to grant welcome credits on first sign-in.
- **`src/middleware.ts`**: Uses NextAuth's `auth()` as middleware. Protects all routes except `/`, `/login`, `/auth/*`, `/api/auth/*`, `/api/stripe/webhook`. Returns 401 JSON for unauthenticated API calls; redirects to `/login` for pages.
- **Session**: JWT tokens carry `user.id`. Client-side access via `SessionProvider`.

---

## Database (PostgreSQL + Drizzle ORM)

The project uses **PostgreSQL 16** with **Drizzle ORM**.

- **Connection**: `postgres-js` driver configured in `src/lib/db/index.ts`.
- **Schema**: All tables defined in `src/lib/db/schema.ts` (8 tables: `user`, `account`, `session`, `verification_token`, `discussion_sessions`, `credits`, `credit_transactions`, `purchases`).
- **Data Access**: Server-side queries in `src/lib/db/*.ts` files (e.g., `credits.ts`, `user-api-key.ts`). All database operations should go through these modules.
- **Migrations**: SQL migration files in `scripts/` (e.g., `init-db.sql`, `migrate-add-api-key.sql`, `migrate-add-credits.sql`). Drizzle migrations in `src/lib/db/migrations/`.
- **Config**: `drizzle.config.ts` at project root.

---

## AI Integration (Vercel AI SDK)

- **SDK**: `ai` + `@ai-sdk/openai` — uses `generateObject()` with Zod schemas for structured output.
- **Models**: Two configurable modes via env vars:
  - `OPENAI_MODEL_NORMAL`: Lighter/cheaper model for standard discussions.
  - `OPENAI_MODEL_DEEP`: Reasoning model for deeper analysis.
- **Base URL**: Configurable via `OPENAI_BASE_URL` for OpenAI-compatible endpoints.
- **Prompts**: All AI prompts for HKDSE discussion simulation are in `src/lib/ai/prompts.ts`.
- **Image OCR**: Uses OpenAI vision model for extracting text from uploaded images.
- **TTS**: Uses OpenAI-compatible TTS API (`tts-1`) with 4 distinct voices (nova, alloy, fable, echo) — one per virtual student.

---

## Credits & Payment System (Stripe)

- **Credits**: Users receive `WELCOME_CREDITS` (default 20) on first sign-in. Each generation costs `GENERATION_COST` (default 10) credits.
- **Own API Key**: Users can bring their own OpenAI API key to bypass credit charges.
- **Payment**: Stripe Checkout (one-time payments) in HKD, supporting card, Alipay, and WeChat Pay.
- **Plans**: Two configurable plans (Plan A/B) with credit amounts and HKD pricing set via env vars.
- **Webhook**: `/api/stripe/webhook` handles `checkout.session.completed` and `checkout.session.expired`.
- **Refund**: Credit refund API for failed generations.

---

## Backend & API Patterns

### 1. API Routes

API routes are in `src/app/api/`. Key endpoints:

| Route | Methods | Purpose |
|-------|---------|---------|
| `/api/auth/[...nextauth]` | GET/POST | NextAuth handler |
| `/api/generate` | POST | Main dialogue generation (file upload via FormData) |
| `/api/tts` | POST | Text-to-speech generation |
| `/api/upload-audio` | POST | Upload audio file |
| `/api/audio/[...path]` | GET | Serve audio files (24h TTL, path traversal protection) |
| `/api/history` | GET/POST | List/save discussion sessions |
| `/api/history/[id]` | * | Individual session operations |
| `/api/stripe/checkout` | POST | Create Stripe Checkout session |
| `/api/stripe/plans` | GET | List available credit plans |
| `/api/stripe/webhook` | POST | Stripe webhook handler |
| `/api/credits/refund` | POST | Refund credits |
| `/api/export/docx` | POST | Generate and download DOCX |
| `/api/user/api-key` | GET/PUT | Manage user's own API key |
| `/api/user/credits` | GET | Get credit balance |
| `/api/user/purchases` | GET | Get purchase history |

### 2. File Processing Pipeline

1. **Client-side** (`pdf-client.ts`): Extracts text layer via pdfjs-dist; if insufficient text, renders pages to canvas images for OCR.
2. **Server-side** (`file-processing/`): PDF via `pdf-parse`, DOCX via `mammoth`, images via OpenAI vision.
3. Files temporarily saved to `tmp/uploads/`, processed, then deleted.

### 3. Environment Variables

- Refer to `.env.example` for all available environment variables (~20 variables).
- **Categories**: AI provider (key, base URL, models), auth (NextAuth + Google OAuth), database (`DATABASE_URL`), Stripe/billing (keys, webhook secret, plan pricing), credits config (welcome credits, generation cost), optional Sentry.
- **Never commit** `.env` or `.env.local` files.

---

## Security & Safety

- **Secrets**: Do not hardcode API keys or credentials.
- **Validation**: Use Zod to validate all external inputs (user input, file uploads, AI responses).
- **Auth Middleware**: All API routes (except auth/webhook) require authentication. All page routes (except landing/login) redirect unauthenticated users.
- **File Serving**: Audio file serving has path traversal protection and 24h TTL.
- **Destructive Actions**: Avoid `rm -rf` or history rewriting in git unless explicitly requested.

---

## Agent Instructions

- **Read First**: Always read the relevant file and its neighbors before proposing edits.
- **Follow Patterns**: If adding a new component, look at existing components in the appropriate `src/components/` subdirectory for reference implementations.
- **Keep it Focused**: Make small, cohesive changes. Avoid unrelated refactors.
- **Validate**: Run `npm run lint` and `npm run build` to ensure your changes don't break the build.
- **Database Changes**: If modifying database schema, update `src/lib/db/schema.ts` and run `npm run db:generate` to create a migration. Also add a SQL migration in `scripts/` following the existing naming convention.
- **API Routes**: New API routes should follow existing patterns — use Zod for input validation, Drizzle ORM for database access, and proper error handling with appropriate HTTP status codes.
- **Communication**: Summarize what changed, where, and why. Call out tradeoffs, assumptions, and known limitations. If validation could not be run, say so explicitly.
- **Clarity**: Prefer clarity and simplicity over cleverness. Preserve existing behavior unless the task explicitly requires changes.
- **UI Consistency**: Ensure all new UI elements support both light and dark modes using Tailwind `dark:` classes.

<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->
