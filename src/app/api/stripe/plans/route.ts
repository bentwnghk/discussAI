import { NextResponse } from "next/server";
import { getCreditPlans } from "@/lib/stripe";
import { getGenerationCost, getResponseCost } from "@/lib/db/credits";

export async function GET() {
  return NextResponse.json({ plans: getCreditPlans(), generationCost: getGenerationCost(), responseCost: getResponseCost() });
}
