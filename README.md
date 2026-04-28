## Mr.🆖 DiscussAI

### __Your AI-Powered HKDSE Oral Discussion Coach__

_Upload. Discuss. Master._

Transform any HKDSE discussion topic into a realistic 4-person group dialogue with AI-generated conversations, text-to-speech audio, and comprehensive learning notes — designed for Hong Kong secondary students preparing for English oral exams.

![Next.js](https://img.shields.io/badge/Next.js_16-111111?style=flat&logo=nextdotjs&logoColor=white)
![React 19](https://img.shields.io/badge/React_19-61DAFB?style=flat&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat&logo=tailwindcss&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-default.svg)

---

## Why Students & Teachers Love Mr.🆖 DiscussAI

| __Realistic__ | __Interactive__ | __Comprehensive__ | __Convenient__ |
| --- | --- | --- | --- |
| Simulates authentic HKDSE group discussions with 4 virtual students | Listen to full audio playback with distinct voices for each candidate | Learning notes with vocabulary, strategies & key ideas in English + Traditional Chinese | Upload PDFs, DOCX, images or type your own topic |

---

## Your Learning Journey

```
flowchart LR
    subgraph INPUT["📷 Input"]
        A[Upload PDF/DOCX/Image]
        B[Type Discussion Topic]
    end
    
    subgraph GENERATE["🤖 AI Generation"]
        C[Extract Topic & Prompts]
        D[Generate 4-Person Dialogue]
        E[Create Learning Notes]
        F[Produce TTS Audio]
    end
    
    subgraph LEARN["📖 Learn"]
        G[Read Transcript]
        H[Listen to Audio]
        I[Review Vocabulary]
        J[Study Strategies]
    end
    
    subgraph EXPORT["📄 Export"]
        K[Download DOCX]
    end
    
    A & B --> C
    C --> D
    D --> E & F
    D --> G
    F --> H
    E --> I & J
    G & I & J --> K
```

---

## Core Features

### 🎤 1. AI Group Discussion Generator

__Turn any topic into a realistic HKDSE oral discussion.__ The AI generates a natural 4-person dialogue (Candidates A–D) that models authentic interaction strategies and runs 6–7 minutes when spoken aloud.

| Feature | Description |
| --- | --- |
| 🗣️ __4 Virtual Students__ | Distinct voices (nova, alloy, fable, echo) via TTS |
| ⏱️ __6–7 Minute Dialogue__ | ~120–150 words per minute, natural conversational pace |
| 🔄 __Two Modes__ | Normal (fast) and Deeper (reasoning model for richer analysis) |
| 📥 __Flexible Input__ | Upload files (PDF, DOCX, images) or type a topic directly |
| 🎯 __Exam-Aligned__ | Follows HKDSE discussion format with question prompts |

---

### 📊 2. Interaction Strategies Modeled

The generated dialogues demonstrate all key HKDSE discussion strategies:

| Strategy | Example |
| --- | --- |
| 🚀 __Initiating__ | "Let's begin by talking about the reasons why..." |
| 🔄 __Maintaining__ | "What do you think? Any thoughts, Candidate C?" |
| 🔀 __Transitioning__ | "Shall we move on and discuss...?" |
| ✅ __Responding__ | "I agree." / "Sorry, I disagree." |
| 🔁 __Rephrasing__ | "I see what you mean. You were saying that..." |
| ❓ __Clarifying__ | "Did you mean that...?" |

---

### 📝 3. Learning Notes

__Comprehensive bilingual notes for every discussion.__

| Section | Content |
| --- | --- |
| 💡 __Ideas__ | Structured outline of main ideas with question prompt references |
| 📖 __Language__ | 12–15 vocabulary words with English definitions, 繁體中文 translations, and usage examples |
| 🗣️ __Strategies__ | 6–8 interaction strategies with examples and 中文 explanations |

---

### 🔊 4. Text-to-Speech Audio

Listen to the full discussion with distinct AI voices for each candidate.

| Feature | Description |
| --- | --- |
| 🎙️ __4 Distinct Voices__ | nova, alloy, fable, echo — one per candidate |
| ▶️ __Audio Player__ | Play, pause, seek with speaker-highlighted transcript sync |
| 💾 __Persistent Storage__ | Audio saved and accessible for 24 hours |

---

### 📷 5. Smart File Processing

Upload any exam paper or reading material and the AI extracts the discussion topic automatically.

| Format | Processing Method |
| --- | --- |
| 📕 __PDF__ | Client-side text extraction + server-side pdf-parse fallback |
| 📗 __DOCX__ | mammoth text extraction |
| 🖼️ __Images__ | OpenAI Vision OCR |

---

### 📜 6. History & Session Management

All your past discussions are saved and accessible anytime.

| Feature | Description |
| --- | --- |
| 📋 __Session List__ | Browse all past discussions with title, date, and mode |
| 🔍 __Full Replay__ | Re-read transcripts and learning notes from any session |
| ▶️ __Audio Replay__ | Listen to saved audio (within TTL) |
| 📤 __Export__ | Download any session as a formatted DOCX document |

---

### 📄 7. DOCX Export

Download your discussions and notes for offline study.

| Export Content | Description |
| --- | --- |
| 🗣️ __Transcript__ | Full dialogue with speaker labels and color coding |
| 📖 __Learning Notes__ | Ideas, vocabulary table, and communication strategies |

---

### 📱 8. Progressive Web App (PWA)

__Install Mr.🆖 DiscussAI on any device for a native app-like experience.__

| Feature | Description |
| --- | --- |
| 📲 __Installable__ | Add to home screen on Android, iOS (Safari), and desktop browsers |
| 🔄 __Offline Caching__ | Serwist-powered precaching with runtime caching strategies |
| 🖼️ __App Icons__ | SVG, 192x192, 512x512 icons + maskable variants for all platforms |
| 🍎 __iOS Support__ | Custom install prompt with step-by-step Safari instructions |
| 🤖 __Android/Desktop__ | Native "Install App" prompt via `react-use-pwa-install` |
| 🔕 __Dismissable__ | Session-persistent dismiss — prompt won't reappear until next visit |

---

## PWA Technical Details

### Architecture

The app is a fully installable PWA built with **Serwist** (a service worker library for Next.js):

```
┌─────────────────────────────────────────────────────┐
│  next.config.ts                                     │
│  └─ withSerwistInit() wraps config at build time    │
│     ├─ swSrc: src/app/sw.ts (service worker source) │
│     └─ swDest: public/sw.js (compiled output)       │
├─────────────────────────────────────────────────────┤
│  src/app/manifest.json                              │
│  └─ Web App Manifest (Next.js metadata convention)  │
│     ├─ display: standalone                          │
│     ├─ orientation: portrait                        │
│     └─ 5 icons: SVG + 192/512 PNG + maskable       │
├─────────────────────────────────────────────────────┤
│  src/app/sw.ts (Service Worker)                     │
│  └─ Serwist instance with:                          │
│     ├─ Precaching (build assets)                    │
│     ├─ Runtime caching (defaultCache strategies)    │
│     ├─ skipWaiting + clientsClaim                   │
│     └─ Navigation preload                           │
├─────────────────────────────────────────────────────┤
│  Client Components                                  │
│  ├─ ServiceWorkerRegistrar                          │
│  │  └─ Registers /sw.js on mount                   │
│  └─ PWAInstallPrompt                                │
│     ├─ Android/Desktop: native install prompt       │
│     └─ iOS: manual instructions (Share → Add)       │
├─────────────────────────────────────────────────────┤
│  providers.tsx                                      │
│  └─ Mounts ServiceWorkerRegistrar + PWAInstallPrompt│
│     at app root (outside auth boundaries)           │
└─────────────────────────────────────────────────────┘
```

### Key Libraries

| Library | Version | Purpose |
| --- | --- |
| `serwist` | ^9.5.7 | Service worker generation and runtime caching |
| `@serwist/next` | ^9.5.7 | Next.js integration (build-time SW compilation) |
| `react-use-pwa-install` | ^1.0.3 | React hook for native `beforeinstallprompt` event |

### Web App Manifest (`src/app/manifest.json`)

- **`display: standalone`** — hides browser UI for app-like experience
- **`orientation: portrait`** — optimized for mobile usage
- **`id: mrng-discussai`** — unique PWA identity
- **Icons**: SVG (`logo.svg`) for scalable display + 192x192/512x512 PNG for all platforms + maskable variants for adaptive icons on Android

### Service Worker (`src/app/sw.ts`)

- **Precaching**: Build assets are precached via Serwist's injection manifest (`__SW_MANIFEST`)
- **Runtime caching**: Uses `defaultCache` from `@serwist/next/worker` with standard strategies (CacheFirst for static assets, NetworkFirst for navigation)
- **Lifecycle**: `skipWaiting` and `clientsClaim` ensure new SW versions activate immediately
- **Navigation preload**: Enabled for faster page loads after SW activation

### Build Integration (`next.config.ts`)

Serwist is conditionally applied **only during production builds** (`PHASE_PRODUCTION_BUILD`) to avoid interfering with development:

```ts
if (phase === PHASE_PRODUCTION_BUILD) {
  const withSerwist = withSerwistInit({
    swSrc: "src/app/sw.ts",
    swDest: "public/sw.js",
    register: false, // Manual registration via ServiceWorkerRegistrar
  });
  return withSerwist(nextConfig);
}
```

Manual registration (`register: false`) is used so the app controls when and how the SW is registered via the `ServiceWorkerRegistrar` component.

### Install Prompt Behavior

| Platform | Behavior |
| --- | --- |
| __Android (Chrome)__ | Native install prompt via `beforeinstallprompt` event; "Install App" button |
| __Desktop (Chrome/Edge)__ | Native install prompt; "Install App" button |
| __iOS (Safari)__ | Custom banner with step-by-step instructions (Share → Add to Home Screen) |
| __Already installed__ | Prompt hidden (detects `display-mode: standalone` or `navigator.standalone`) |
| __Dismissed__ | Hidden for current session via `sessionStorage` |

### Middleware Exception

`manifest.json` is excluded from auth middleware to allow the browser to fetch it unauthenticated:

```ts
matcher: ["/((?!_next/static|_next/image|favicon.ico|logo.png|icon.png|manifest.json|examples).*)"]
```

---

## Credits & Payment System

| Aspect | Details |
| --- | --- |
| 🎁 __Welcome Credits__ | 20 free credits on first sign-in |
| 💰 __Cost per Generation__ | 10 credits per discussion |
| 🔑 __Own API Key__ | Bring your own OpenAI-compatible key to bypass credit charges |
| 💳 __Stripe Payments__ | One-time purchases in HKD (card, Alipay, WeChat Pay) |
| 📊 __Two Plans__ | Configurable Plan A & Plan B with different credit amounts |

---

## AI Models Supported

| Provider | Models |
| --- | --- |
| 🟢 __Google Gemini__ | gemini-3-flash |
| 🔵 __OpenAI__ | gpt-5.1, gpt-5-mini |
| 🟣 __DeepSeek__ | deepseek-chat |

### Access Modes

| Mode | Description |
| --- | --- |
| 💰 __Credits Mode__ | Use built-in credits for generation |
| 🔑 __Own API Key__ | Bring your own OpenAI-compatible API key |

---

## Languages

| Language | Code | Status |
| --- | --- | --- |
| 🇬🇧 English | en-US | ✅ Full Support |
| 🇭🇰 繁體中文 | zh-HK | ✅ Full Support (learning notes) |

---

## Tech Stack

| Category | Technology |
| --- | --- |
| ⚡ __Framework__ | Next.js 16 (App Router) |
| ⚛️ __UI Library__ | React 19 |
| 🔷 __Language__ | TypeScript |
| 🎨 __Styling__ | Tailwind CSS v4 + Shadcn UI |
| 🗄️ __Database__ | PostgreSQL 16 + Drizzle ORM |
| 🔐 __Auth__ | NextAuth v5 (Google OAuth, JWT) |
| 🤖 __AI__ | Vercel AI SDK + @ai-sdk/openai |
| 💳 __Payments__ | Stripe |
| 🎭 __Icons__ | Lucide React |
| 🎞️ __Animations__ | Motion (Framer Motion) |
| 🔊 __TTS__ | OpenAI-compatible Speech API |
| 📱 __PWA__ | Serwist (service worker + caching) |
| 📄 __Document Export__ | docx |

---

## Getting Started

### Prerequisites

- Node.js >= 18
- PostgreSQL 16
- Google OAuth credentials
- OpenAI-compatible API key

### Installation

```bash
# Clone the repository
git clone https://github.com/bentwnghk/discussAI.git
cd discussAI

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Set up the database
npm run db:push

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the app.

### Docker

```bash
docker compose up --build
```

---

## Project Structure

```
src/
├── middleware.ts                # Auth middleware (route protection, redirects)
├── app/                        # Next.js App Router
│   ├── manifest.json           # PWA Web App Manifest
│   ├── sw.ts                   # Service worker source (Serwist)
│   ├── (app)/                  # Authenticated routes
│   │   ├── discuss/            # Main discussion generation page
│   │   ├── history/            # Session history + detail view
│   │   └── credits/            # Credit purchase page
│   ├── (auth)/                 # Unauthenticated routes (login, error)
│   └── api/                    # API route handlers
├── components/
│   ├── pwa-install-prompt.tsx    # PWA install prompt (Android/iOS/Desktop)
│   ├── service-worker-registrar.tsx # Service worker registration
│   ├── providers.tsx             # Client providers (Session, PWA, Toaster)
│   ├── discuss/                # Core discussion feature components
│   │   ├── audio-player.tsx    # Audio playback with transcript sync
│   │   ├── file-upload.tsx     # File upload (PDF, DOCX, images)
│   │   ├── learning-notes.tsx  # Learning notes display
│   │   └── transcript-display.tsx # Dialogue transcript renderer
│   ├── landing/                # Landing/marketing page
│   ├── layout/                 # Header, footer
│   └── ui/                     # Shadcn UI primitives
├── hooks/                      # React Context hooks (API key, credits)
├── lib/
│   ├── ai/                     # AI dialogue generation + prompts + schemas
│   ├── db/                     # Drizzle ORM setup + schema + queries
│   ├── file-processing/        # PDF, DOCX, image OCR processing
│   ├── tts/                    # Text-to-speech generation
│   ├── export/                 # DOCX export
│   ├── auth.ts                 # NextAuth v5 configuration
│   └── stripe.ts               # Stripe client + plan definitions
└── types/                      # TypeScript types + speaker mappings
```

---

## API Routes

| Route | Methods | Purpose |
| --- | --- | --- |
| `/api/generate` | POST | Generate dialogue from topic/file |
| `/api/tts` | POST | Text-to-speech generation |
| `/api/upload-audio` | POST | Upload audio file |
| `/api/audio/[...path]` | GET | Serve audio files (24h TTL) |
| `/api/history` | GET/POST | List/save discussion sessions |
| `/api/history/[id]` | GET/DELETE | Individual session operations |
| `/api/export/docx` | POST | Generate DOCX download |
| `/api/stripe/checkout` | POST | Create Stripe Checkout session |
| `/api/stripe/plans` | GET | List available credit plans |
| `/api/stripe/webhook` | POST | Stripe webhook handler |
| `/api/credits/refund` | POST | Refund credits |
| `/api/user/api-key` | GET/PUT | Manage user API key |
| `/api/user/credits` | GET | Get credit balance |
| `/api/user/purchases` | GET | Get purchase history |

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | AI provider API key |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint URL |
| `OPENAI_MODEL_NORMAL` | Model for standard discussions |
| `OPENAI_MODEL_DEEP` | Model for deeper analysis |
| `DATABASE_URL` | PostgreSQL connection string |
| `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` | Google OAuth credentials |
| `STRIPE_SECRET_KEY` | Stripe API key |

---

## License

MIT License - Free for personal use.

---

## Acknowledgments

- Shadcn UI for beautiful components
- Vercel AI SDK for AI integration
- Drizzle ORM for type-safe database access
- All the amazing open-source libraries that make this possible
