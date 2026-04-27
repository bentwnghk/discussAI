import { NextRequest, NextResponse } from "next/server";
import { getStripe, getPlanByKey } from "@/lib/stripe";
import { addCreditsFromPurchase, markPurchaseFailed } from "@/lib/db/credits";
import crypto from "crypto";

export async function POST(req: NextRequest) {
  const body = await req.text();
  const signature = req.headers.get("stripe-signature");
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  console.log("[webhook] sig header:", signature?.slice(0, 80));
  console.log("[webhook] secret prefix:", webhookSecret?.slice(0, 14));
  console.log("[webhook] secret length:", webhookSecret?.length);
  console.log("[webhook] body length:", body.length, "first 80:", body.slice(0, 80));
  console.log("[webhook] body has CRLF:", body.includes('\r\n'));
  console.log("[webhook] body hex[0..20]:", Buffer.from(body.slice(0, 20)).toString('hex'));

  // manually compute the expected HMAC to isolate body vs secret mismatch
  if (signature && webhookSecret) {
    const timestamp = signature.split(',')[0]?.split('=')[1];
    const expectedV1 = signature.split(',')[1]?.split('=')[1];
    const computed = crypto.createHmac('sha256', webhookSecret)
      .update(`${timestamp}.${body}`, 'utf8')
      .digest('hex');
    console.log("[webhook] expected v1  :", expectedV1);
    console.log("[webhook] computed hmac:", computed);
    console.log("[webhook] hmac match:", computed === expectedV1);
  }

  if (!signature || !webhookSecret) {
    return NextResponse.json({ error: "Missing signature or secret" }, { status: 400 });
  }

  let event;
  try {
    const stripe = getStripe();
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Invalid signature";
    console.error("Webhook signature verification failed:", message);
    return NextResponse.json({ error: message }, { status: 400 });
  }

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object;
        const userId = session.metadata?.userId;
        const planKey = session.metadata?.planKey;
        const credits = parseInt(session.metadata?.credits || "0", 10);
        const amountHKD = parseFloat(session.metadata?.amountHKD || "0");

        if (!userId || !credits) {
          console.error("Missing metadata in checkout session:", session.id);
          break;
        }

        await addCreditsFromPurchase(
          userId,
          credits,
          session.id,
          session.payment_intent as string | null,
          planKey ? (getPlanByKey(planKey)?.label || planKey) : "unknown",
          amountHKD
        );
        break;
      }
      case "checkout.session.expired": {
        const session = event.data.object;
        await markPurchaseFailed(session.id);
        break;
      }
      default:
        console.log(`Unhandled event type: ${event.type}`);
    }
  } catch (error) {
    console.error("Webhook handler error:", error);
    return NextResponse.json({ error: "Webhook handler failed" }, { status: 500 });
  }

  return NextResponse.json({ received: true });
}
