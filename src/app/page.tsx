import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Rocket, Star, Coins } from "lucide-react";
import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import { getCreditPlans } from "@/lib/stripe";
import { getWelcomeCredits } from "@/lib/db/credits";

export default async function HomePage() {
  const session = await auth();
  if (session) redirect("/discuss");

  const plans = getCreditPlans();
  const welcomeCredits = getWelcomeCredits();

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

        <div className="mt-12">
          <div className="flex items-center justify-center gap-2 mb-6">
            <Coins className="h-5 w-5" />
            <span className="text-lg font-semibold">
              Sign up and get {welcomeCredits} free credits ({Math.floor(welcomeCredits / 10)} free discussions!)
            </span>
          </div>
          <div className="grid gap-6 sm:grid-cols-2 max-w-2xl mx-auto">
            {plans.map((plan) => (
              <Card
                key={plan.key}
                className={`relative ${plan.highlight ? "border-primary shadow-lg" : ""}`}
              >
                {plan.highlight && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <Badge className="bg-primary text-primary-foreground px-3 py-1">
                      <Star className="mr-1 h-3 w-3" />
                      Best Value
                    </Badge>
                  </div>
                )}
                <CardHeader className="text-center pb-2">
                  <CardTitle className="text-xl">{plan.label}</CardTitle>
                  <CardDescription>{plan.credits} Credits</CardDescription>
                </CardHeader>
                <CardContent className="text-center">
                  <span className="text-4xl font-bold">HK${plan.priceHKD}</span>
                  {plan.highlight && (
                    <p className="text-sm text-muted-foreground mt-2">
                      ~HK${(plan.priceHKD / plan.credits * 10).toFixed(1)} per discussion
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
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
