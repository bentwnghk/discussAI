import { LandingPage } from "@/components/landing/landing-page";
import { auth } from "@/lib/auth";
import { redirect } from "next/navigation";
import { getWelcomeCredits, getGenerationCost } from "@/lib/db/credits";

export default async function HomePage() {
  const session = await auth();
  if (session) redirect("/discuss");

  const welcomeCredits = getWelcomeCredits();
  const generationCost = getGenerationCost();

  return (
    <LandingPage
      welcomeCredits={welcomeCredits}
      generationCost={generationCost}
    />
  );
}
