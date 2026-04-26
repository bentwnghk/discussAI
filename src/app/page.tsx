import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Rocket, Star, Coins, Zap, Package } from "lucide-react";
import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import { getCreditPlans } from "@/lib/stripe";
import { getWelcomeCredits, getGenerationCost } from "@/lib/db/credits";

export default async function HomePage() {
  const session = await auth();
  if (session) redirect("/discuss");

  const plans = getCreditPlans();
  const welcomeCredits = getWelcomeCredits();
  const generationCost = getGenerationCost();

  return (
    <div className="container mx-auto px-4 py-16">
      <div className="mx-auto max-w-3xl text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
          Mr.🆖 DiscussAI
        </h1>
        <p className="mt-4 text-lg text-muted-foreground">
          The ultimate HKDSE Speaking simulator
        </p>
        <p className="mt-6 text-base leading-relaxed text-muted-foreground">
          Speak like a 5**—anytime, anywhere!
          <br />
          Don't just practice; Simulate. Transform any discussion topic into
          a high-stakes HKDSE discussion featuring natural voices and
          exam-specific vocabulary. Master the art of authentic interaction
          and turn-taking with strategies modeled after real exam success.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 max-w-2xl mx-auto text-left">
          {[
            { emoji: "📁", title: "Snap, Upload, Speak", desc: "Seamlessly upload images, PDFs, or DOCX files to generate custom speaking simulations on the fly." },
            { emoji: "🗣️", title: "Authentic Group Dynamics", desc: "Engage in 4-student interactions featuring realistic voices and turn-taking strategies modeled after the HKDSE Paper 4 format." },
            { emoji: "📚", title: "Comprehensive Smart Notes", desc: "Gain a competitive edge with structured learning notes, advanced vocabulary, and tactical interaction tips." },
            { emoji: "📄", title: "Study Your Way—Online or Off", desc: "Learn seamlessly in-app or download a hard copy to highlight, annotate, and study distraction-free." },
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
          <div className="grid gap-6 sm:grid-cols-2 max-w-2xl mx-auto pt-4">
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
                <Card
                  key={plan.key}
                  className={`relative pt-6 ${plan.highlight ? "border-primary shadow-lg !overflow-visible" : ""}`}
                >
                  {plan.highlight && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                      <Badge className="bg-primary text-primary-foreground px-3 py-1">
                        <Star className="mr-1 h-3 w-3" />
                        Best Value
                      </Badge>
                    </div>
                  )}
                  <CardHeader className="text-center pb-2 items-center">
                    <div className="mb-2">
                      {plan.highlight ? (
                        <Zap className="h-8 w-8 text-primary" />
                      ) : (
                        <Package className="h-8 w-8 text-muted-foreground" />
                      )}
                    </div>
                    <CardTitle className="text-xl">{plan.label}</CardTitle>
                    <CardDescription>{plan.credits} Credits</CardDescription>
                  </CardHeader>
                  <CardContent className="text-center space-y-2">
                    <span className="text-4xl font-bold">HK${plan.priceHKD}</span>
                    <p className="text-sm text-muted-foreground">
                      {discussions} discussions (~HK${(plan.priceHKD / discussions).toFixed(1)} each)
                    </p>
                    {plan.highlight && savedPct > 0 && (
                      <p className="text-sm font-medium text-primary">
                        Save {savedPct}% compared to Starter
                      </p>
                    )}
                  </CardContent>
                </Card>
              );
            })}
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
