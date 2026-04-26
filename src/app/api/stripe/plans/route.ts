import { NextResponse } from "next/server";
import { getCreditPlans } from "@/lib/stripe";

export async function GET() {
  return NextResponse.json({ plans: getCreditPlans() });
}
