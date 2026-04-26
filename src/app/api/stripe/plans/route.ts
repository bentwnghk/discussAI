import { NextResponse } from "next/server";
import { getCreditPlans } from "@/lib/stripe";
import { getGenerationCost } from "@/lib/db/credits";

export async function GET() {
  return NextResponse.json({ plans: getCreditPlans(), generationCost: getGenerationCost() });
}
