"use client";

import { useRef } from "react";
import Link from "next/link";
import { motion, useInView } from "motion/react";
import {
  Upload,
  BookOpen,
  FileText,
  Rocket,
  Coins,
  Star,
  Zap,
  Package,
  Users,
  Headphones,
  Brain,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Download,
  Volume2,
  Target,
  History,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const easeOut: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

const sectionVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12 } },
};

const cardVariants = {
  hidden: { opacity: 0, y: 40, scale: 0.95 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.6, ease: easeOut },
  },
};

const heroItemVariants = {
  hidden: { opacity: 0, y: 30, filter: "blur(10px)" },
  visible: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.7, ease: easeOut },
  },
};

const sectionTitleVariants = {
  hidden: { opacity: 0, x: -30 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6 } },
};

const pillVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.4 } },
};

function AnimatedSection({
  children,
  className,
  staggerDelay = 0.12,
}: {
  children: React.ReactNode;
  className?: string;
  staggerDelay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });
  return (
    <motion.div
      ref={ref}
      variants={gridContainer(staggerDelay)}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      className={className}
    >
      {children}
    </motion.div>
  );
}

function gridContainer(stagger: number) {
  return {
    hidden: {},
    visible: { transition: { staggerChildren: stagger } },
  };
}

function seededRandom(seed: number) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function generateGalaxyData() {
  const stars: { x: number; y: number; r: number; o: number }[] = [];
  for (let i = 0; i < 120; i++) {
    stars.push({
      x: seededRandom(i * 2) * 1000,
      y: seededRandom(i * 2 + 1) * 1000,
      r: seededRandom(i * 3) * 2.5 + 0.8,
      o: seededRandom(i * 5) * 0.5 + 0.15,
    });
  }
  const lines: { x1: number; y1: number; x2: number; y2: number; o: number }[] = [];
  const maxDist = 180;
  for (let i = 0; i < stars.length; i++) {
    for (let j = i + 1; j < stars.length; j++) {
      const dx = stars[i].x - stars[j].x;
      const dy = stars[i].y - stars[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < maxDist) {
        lines.push({
          x1: stars[i].x,
          y1: stars[i].y,
          x2: stars[j].x,
          y2: stars[j].y,
          o: (1 - d / maxDist) * 0.25,
        });
      }
    }
  }
  return { stars, lines };
}

const galaxyData = generateGalaxyData();

function GalaxyBackground() {
  const { stars, lines } = galaxyData;

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <svg
        viewBox="0 0 1000 1000"
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="xMidYMid slice"
      >
        {lines.map((l, i) => (
          <line
            key={`l-${i}`}
            x1={l.x1}
            y1={l.y1}
            x2={l.x2}
            y2={l.y2}
            stroke="currentColor"
            className="text-white/20 dark:text-white/10"
            strokeWidth="0.5"
            opacity={l.o}
          />
        ))}
        {stars.map((s, i) => (
          <circle
            key={`s-${i}`}
            cx={s.x}
            cy={s.y}
            r={s.r}
            className="text-white/60 dark:text-white/40"
            fill="currentColor"
            opacity={s.o}
          />
        ))}
      </svg>
    </div>
  );
}

function WaveDivider({ flip = false }: { flip?: boolean }) {
  return (
    <div className={`w-full overflow-hidden leading-[0] ${flip ? "rotate-180" : ""}`}>
      <svg
        viewBox="0 0 1440 100"
        preserveAspectRatio="none"
        className="relative block w-full h-[60px]"
      >
        <path
          d="M0,40 C360,100 720,0 1080,60 C1260,80 1380,40 1440,50 L1440,100 L0,100 Z"
          className="fill-background"
        />
      </svg>
    </div>
  );
}

const glassBase =
  "bg-white/[0.08] dark:bg-white/[0.04] backdrop-blur-xl border border-white/[0.15] dark:border-white/[0.08] shadow-[inset_0_0_20px_rgba(255,255,255,0.05)]";
const glassCard = `${glassBase} bg-emerald-50/[0.06] dark:bg-emerald-900/[0.03]`;

function GoogleIcon() {
  return (
    <svg className="mr-2 h-5 w-5" viewBox="0 0 24 24">
      <path
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
        fill="#4285F4"
      />
      <path
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
        fill="#34A853"
      />
      <path
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
        fill="#FBBC05"
      />
      <path
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
        fill="#EA4335"
      />
    </svg>
  );
}

interface CreditPlan {
  key: string;
  label: string;
  credits: number;
  priceHKD: number;
  highlight?: boolean;
}

interface LandingPageProps {
  plans: CreditPlan[];
  welcomeCredits: number;
  generationCost: number;
}

export function LandingPage({ plans, welcomeCredits, generationCost }: LandingPageProps) {
  const features = [
    { icon: Upload, key: "upload", color: "text-blue-600 dark:text-blue-400", title: "Snap, Upload, Speak", desc: "Seamlessly upload images, PDFs, or DOCX files to generate custom speaking simulations on the fly." },
    { icon: Users, key: "group", color: "text-emerald-600 dark:text-emerald-400", title: "Authentic Group Dynamics", desc: "Engage in 4-student interactions featuring realistic voices and turn-taking strategies modeled after HKDSE Paper 4." },
    { icon: Volume2, key: "tts", color: "text-teal-600 dark:text-teal-400", title: "Natural Voices", desc: "Listen to realistic AI-generated speech with authentic accents and exam-appropriate intonation." },
    { icon: BookOpen, key: "notes", color: "text-purple-600 dark:text-purple-400", title: "Comprehensive Smart Notes", desc: "Gain a competitive edge with structured learning notes, advanced vocabulary, and tactical interaction tips." },
    { icon: Target, key: "exam", color: "text-rose-600 dark:text-rose-400", title: "Exam-Specific Vocabulary", desc: "Master the exact phrases, discourse markers, and expressions that HKDSE examiners look for." },
    { icon: Sparkles, key: "adaptive", color: "text-pink-600 dark:text-pink-400", title: "Adaptive Difficulty", desc: "Discussions adapt to your level, ensuring you're always challenged but never overwhelmed." },
    { icon: History, key: "history", color: "text-indigo-600 dark:text-indigo-400", title: "Practice History", desc: "Review past discussions with color-coded transcripts, audio playback, learning notes, and one-click Word export." },
    { icon: Download, key: "export", color: "text-amber-600 dark:text-amber-400", title: "Study Your Way", desc: "Learn seamlessly in-app or download a hard copy to highlight, annotate, and study distraction-free." },
  ];

  const journey = [
    { num: 1, icon: Upload, key: "upload", label: "Upload Topic" },
    { num: 2, icon: FileText, key: "extract", label: "Extract & Analyze" },
    { num: 3, icon: Brain, key: "generate", label: "AI Generates Discussion" },
    { num: 4, icon: Headphones, key: "listen", label: "Listen to Discussion" },
    { num: 5, icon: FileText, key: "transcript", label: "Read the Transcript" },
    { num: 6, icon: BookOpen, key: "notes", label: "Review Smart Notes" },
    { num: 7, icon: Download, key: "export", label: "Export & Study" },
  ];

  const skills = [
    "Agreeing & Disagreeing",
    "Turn-Taking",
    "Clarification",
    "Elaboration",
    "Summarizing",
    "Persuasion",
    "Active Listening",
    "Discourse Markers",
  ];

  return (
    <div className="relative">
      <section className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden bg-gradient-to-b from-[#0f172a] via-[#1e293b] to-[#0f172a]">
        <GalaxyBackground />
        <motion.div
          className="relative z-10 container mx-auto px-4 py-20 text-center"
          variants={sectionVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={heroItemVariants}>
            <Badge
              variant="outline"
              className="mb-6 border-white/20 text-white/80 bg-white/5 backdrop-blur-sm px-4 py-1.5"
            >
              <Sparkles className="mr-1 h-3 w-3" />
              The ultimate HKDSE Paper 4 simulator
            </Badge>
          </motion.div>

          <motion.h1
            variants={heroItemVariants}
            className="text-6xl sm:text-8xl font-extrabold tracking-tighter text-white"
          >
            Mr.
            <span className="inline-block mx-1">🆖</span>
            <span className="bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent">
              DiscussAI
            </span>
          </motion.h1>

          <motion.p
            variants={heroItemVariants}
            className="mt-3 text-2xl sm:text-3xl font-medium italic text-white/60 bg-gradient-to-r from-emerald-300/80 via-teal-300/80 to-cyan-300/80 bg-clip-text"
          >
            Speak like a 5** — anytime, anywhere.
          </motion.p>

          <motion.p
            variants={heroItemVariants}
            className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-white/50"
          >
            Don&apos;t just practice; Simulate. Transform any discussion topic into
            a high-stakes HKDSE discussion featuring natural voices and
            exam-specific vocabulary. Master the art of authentic interaction
            and turn-taking with strategies modeled after real exam success.
          </motion.p>

          <motion.div variants={heroItemVariants} className="mt-10">
            <Link href="/login">
              <Button
                size="lg"
                className="h-12 px-8 text-base bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white border-0 shadow-lg shadow-emerald-500/25 cursor-pointer"
              >
                <Rocket className="mr-2 h-5 w-5" />
                Get Started Free
              </Button>
            </Link>
          </motion.div>

          <motion.div
            variants={heroItemVariants}
            className="mt-6 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 backdrop-blur-sm text-white/80 text-sm"
          >
            <Coins className="h-4 w-4 text-amber-400" />
            <span>
              {welcomeCredits} free credits on sign up — that's {Math.floor(welcomeCredits / generationCost)} full discussions to get started!
            </span>
          </motion.div>
        </motion.div>
      </section>

      <WaveDivider />

      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3"
            >
              Features
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Powerful Features
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              Everything you need to ace HKDSE Paper 4, powered by cutting-edge AI.
            </motion.p>
          </AnimatedSection>

          <AnimatedSection className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 max-w-6xl mx-auto">
            {features.map(({ icon: Icon, key, color, title, desc }) => (
              <motion.div
                key={key}
                variants={cardVariants}
                className={`rounded-2xl p-5 ${glassCard} group hover:scale-[1.02] transition-transform duration-300`}
              >
                <div className={`mb-3 p-2.5 rounded-xl bg-white/10 inline-block ${color}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="font-semibold text-base mb-1">{title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{desc}</p>
              </motion.div>
            ))}
          </AnimatedSection>
        </div>
      </section>

      <section className="py-20 px-4">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3"
            >
              How It Works
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Your Discussion Journey
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-12"
            />
          </AnimatedSection>

          <AnimatedSection
            className="max-w-3xl mx-auto space-y-0"
            staggerDelay={0.1}
          >
            {journey.map(({ num, icon: Icon, key, label }, idx) => (
              <motion.div
                key={key}
                variants={cardVariants}
                className="relative flex items-center gap-4"
              >
                <div className="relative flex flex-col items-center">
                  <div className={`flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold text-white shrink-0 ${idx === 0 ? "bg-gradient-to-br from-emerald-500 to-teal-500" : "bg-gradient-to-br from-slate-500 to-slate-600"}`}>
                    {num.toString().padStart(2, "0")}
                  </div>
                  {idx < journey.length - 1 && (
                    <div className="w-0.5 h-12 bg-gradient-to-b from-border to-transparent" />
                  )}
                </div>
                <div className={`flex items-center gap-3 ${idx < journey.length - 1 ? "pb-12" : ""}`}>
                  <div className="p-2 rounded-lg bg-muted">
                    <Icon className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <span className="font-medium text-base">{label}</span>
                </div>
              </motion.div>
            ))}
          </AnimatedSection>
        </div>
      </section>

      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3"
            >
              Skills
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Skills You&apos;ll Master
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-12"
            />
          </AnimatedSection>

          <AnimatedSection className="flex flex-wrap justify-center gap-3 max-w-3xl mx-auto">
            {skills.map((skill) => (
              <motion.span
                key={skill}
                variants={pillVariants}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium bg-white/60 dark:bg-white/10 border border-white/30 dark:border-white/10 backdrop-blur-sm"
              >
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                {skill}
              </motion.span>
            ))}
          </AnimatedSection>
        </div>
      </section>

      <section className="py-20 px-4">
        <div className="container mx-auto">
          <AnimatedSection>
            <motion.span
              variants={sectionTitleVariants}
              className="block text-center text-xs font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-3"
            >
              Pricing
            </motion.span>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-3xl sm:text-5xl font-bold tracking-tight text-center mb-4"
            >
              Simple, Transparent Pricing
            </motion.h2>
            <motion.div
              variants={sectionTitleVariants}
              className="w-24 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 mx-auto rounded-full mb-4"
            />
            <motion.p
              variants={sectionTitleVariants}
              className="text-center text-muted-foreground mb-12 max-w-xl mx-auto"
            >
              Start free, upgrade when you&apos;re ready. Every package gives you full access to all features.
            </motion.p>
          </AnimatedSection>

          <AnimatedSection className="grid gap-6 sm:grid-cols-2 max-w-2xl mx-auto">
            {plans.map((plan) => {
              const discussions = Math.floor(plan.credits / generationCost);
              const starterPlan = plans.find((p) => !p.highlight);
              const savedPct = starterPlan
                ? Math.round(
                    (1 -
                      plan.priceHKD /
                        (plan.credits *
                          starterPlan.priceHKD /
                          starterPlan.credits)) *
                      100
                  )
                : 0;

              return (
                <motion.div
                  key={plan.key}
                  variants={cardVariants}
                  className={`relative rounded-2xl p-6 ${glassCard} ${
                    plan.highlight
                      ? "ring-2 ring-emerald-500/50 shadow-lg shadow-emerald-500/10"
                      : ""
                  }`}
                >
                  {plan.highlight && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                      <Badge className="bg-gradient-to-r from-emerald-500 to-teal-500 text-white border-0 px-3 py-1">
                        <Star className="mr-1 h-3 w-3" />
                        Best Value
                      </Badge>
                    </div>
                  )}
                  <div className="text-center space-y-3">
                    <div className="flex justify-center">
                      {plan.highlight ? (
                        <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                          <Zap className="h-6 w-6" />
                        </div>
                      ) : (
                        <div className="p-2 rounded-xl bg-white/10 text-muted-foreground">
                          <Package className="h-6 w-6" />
                        </div>
                      )}
                    </div>
                    <h3 className="text-xl font-semibold">{plan.label}</h3>
                    <p className="text-sm text-muted-foreground">{plan.credits} Credits</p>
                    <div className="text-5xl font-extrabold tracking-tight">
                      <span className="text-lg align-top">HK$</span>
                      {plan.priceHKD}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {discussions} discussions (~HK${(plan.priceHKD / discussions).toFixed(1)} each)
                    </p>
                    {plan.highlight && savedPct > 0 && (
                      <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
                        Save {savedPct}% compared to Starter
                      </p>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </AnimatedSection>
        </div>
      </section>

      <section className="relative py-24 px-4 overflow-hidden bg-gradient-to-b from-[#0f172a] via-[#1e293b] to-[#0f172a]">
        <GalaxyBackground />
        <div className="relative z-10 container mx-auto text-center">
          <AnimatedSection>
            <motion.h2
              variants={sectionTitleVariants}
              className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white mb-4"
            >
              Ready to Ace HKDSE Paper 4?
            </motion.h2>
            <motion.p
              variants={sectionTitleVariants}
              className="text-white/50 mb-8 max-w-xl mx-auto"
            >
              Join thousands of students who are already practicing smarter, not harder.
            </motion.p>
            <motion.div variants={heroItemVariants}>
              <Link href="/login">
                <Button
                  size="lg"
                  className="h-12 px-8 text-base bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white border-0 shadow-lg shadow-emerald-500/25 cursor-pointer"
                >
                  <GoogleIcon />
                  Sign in with Google
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </motion.div>
          </AnimatedSection>
        </div>
      </section>
    </div>
  );
}
