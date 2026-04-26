import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Rocket } from "lucide-react";
import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";

export default async function HomePage() {
  const session = await auth();
  if (session) redirect("/discuss");

  return (
    <div className="container mx-auto px-4 py-16">
      <div className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
          Mr.🆖 DiscussAI
        </h1>
        <p className="mt-4 text-lg text-muted-foreground">
          AI Group Discussion Tutor for Hong Kong Students
        </p>
        <p className="mt-6 text-base leading-relaxed text-muted-foreground">
          Enhance speaking skills through AI-generated group discussions.
          Transform any Paper 4 questions into realistic audio discussions with authentic
          conversation strategies for HKDSE speaking exam preparation.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 max-w-2xl mx-auto text-left">
          {[
            { emoji: "📁", title: "Upload & Go", desc: "Upload images, PDF, or DOCX of your speaking exam papers" },
            { emoji: "🗣️", title: "4-Student Interaction", desc: "Realistic conversations with 4 distinct AI voices" },
            { emoji: "📚", title: "Study Notes", desc: "Comprehensive learning notes with useful ideas, vocabulary & strategies" },
            { emoji: "📄", title: "DOCX Export", desc: "Download Word documents for offline study or classroom use" },
          ].map((f) => (
            <div key={f.title} className="rounded-lg border p-4">
              <div className="text-2xl mb-2">{f.emoji}</div>
              <h3 className="font-semibold">{f.title}</h3>
              <p className="text-sm text-muted-foreground">{f.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-10 flex justify-center gap-4">
          <Link href="/login">
            <Button size="lg">
              <Rocket className="mr-2 h-5 w-5" />
              Get Started
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
