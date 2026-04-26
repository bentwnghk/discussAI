"use client";

import { useState, useEffect, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useCredits } from "@/hooks/use-credits";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Coins, CheckCircle, XCircle, Loader2, Star, ShoppingCart, MessageSquareText, Zap, Package } from "lucide-react";
import Link from "next/link";

interface PlanConfig {
  key: string;
  label: string;
  credits: number;
  priceHKD: number;
  highlight?: boolean;
}

interface PurchaseRecord {
  id: string;
  planName: string;
  creditsAmount: number;
  amountHKD: number;
  status: string;
  createdAt: string;
}

export default function CreditsPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { balance, refreshBalance } = useCredits();
  const [plans, setPlans] = useState<PlanConfig[]>([]);
  const [generationCost, setGenerationCost] = useState(10);
  const [loading, setLoading] = useState<string | null>(null);
  const [purchases, setPurchases] = useState<PurchaseRecord[]>([]);

  const isSuccess = searchParams.get("success") === "true";
  const isCanceled = searchParams.get("canceled") === "true";

  useEffect(() => {
    if (isSuccess) {
      refreshBalance();
      const timer = setTimeout(() => {
        router.replace("/credits");
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [isSuccess, refreshBalance, router]);

  useEffect(() => {
    fetch("/api/stripe/plans")
      .then((res) => res.json())
      .then((data) => {
        setPlans(data.plans || []);
        if (data.generationCost) setGenerationCost(data.generationCost);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/user/purchases")
      .then((res) => res.json())
      .then((data) => setPurchases(data.purchases || []))
      .catch(() => {});
  }, []);

  const handlePurchase = useCallback(async (planKey: string) => {
    setLoading(planKey);
    try {
      const res = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ planKey }),
      });
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        throw new Error(data.error || "Failed to create checkout session");
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : "Purchase failed");
    } finally {
      setLoading(null);
    }
  }, []);

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold">Credits</h1>
        <p className="text-muted-foreground mt-2">
          Purchase credits to generate discussions
        </p>
      </div>

      {isSuccess && (
        <div className="mb-6 rounded-lg border border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950 p-4 flex items-center gap-3">
          <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
          <div>
            <p className="font-medium text-green-800 dark:text-green-200">
              Payment successful!
            </p>
            <p className="text-sm text-green-700 dark:text-green-300">
              Your credits have been added to your account.
            </p>
          </div>
        </div>
      )}

      {isCanceled && (
        <div className="mb-6 rounded-lg border border-yellow-200 bg-yellow-50 dark:border-yellow-900 dark:bg-yellow-950 p-4 flex items-center gap-3">
          <XCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
          <p className="text-sm text-yellow-700 dark:text-yellow-300">
            Payment was canceled. No charges were made.
          </p>
        </div>
      )}

      <div className="flex items-center justify-center gap-2 mb-8 p-4 rounded-lg bg-muted">
        <Coins className="h-5 w-5" />
        <span className="text-lg font-semibold">
          {balance !== null ? balance : "..."} Credits
        </span>
        <span className="text-muted-foreground">remaining</span>
      </div>

      <div className="grid gap-6 md:grid-cols-2 mb-8 pt-4">
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
              className={`relative pt-6 ${
                plan.highlight
                  ? "border-primary shadow-lg scale-[1.02] !overflow-visible"
                  : ""
              }`}
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
              <CardContent className="text-center space-y-4">
                <div>
                  <span className="text-4xl font-bold">
                    HK${plan.priceHKD}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  {discussions} discussions (~HK${(plan.priceHKD / discussions).toFixed(1)} each)
                </p>
                {plan.highlight && savedPct > 0 && (
                  <p className="text-sm font-medium text-primary">
                    Save {savedPct}% compared to Starter
                  </p>
                )}
                <Button
                  className="w-full"
                  variant={plan.highlight ? "default" : "outline"}
                  size="lg"
                  onClick={() => handlePurchase(plan.key)}
                  disabled={loading !== null}
                >
                  {loading === plan.key ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Redirecting...
                    </>
                  ) : (
                    <>
                      <ShoppingCart className="mr-2 h-4 w-4" />
                      Buy {plan.credits} Credits
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {purchases.length > 0 && (
        <>
          <Separator className="my-8" />
          <div>
            <h2 className="text-xl font-semibold mb-4">Purchase History</h2>
            <div className="rounded-md border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="p-3 text-left font-medium">Date</th>
                    <th className="p-3 text-left font-medium">Package</th>
                    <th className="p-3 text-right font-medium">Amount</th>
                    <th className="p-3 text-right font-medium">Credits</th>
                    <th className="p-3 text-right font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {purchases.map((p) => (
                    <tr key={p.id} className="border-b last:border-0">
                      <td className="p-3">
                        {new Date(p.createdAt).toLocaleDateString("en-HK", {
                          timeZone: "Asia/Hong_Kong",
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })}
                      </td>
                      <td className="p-3">
                        {p.planName}
                      </td>
                      <td className="p-3 text-right">HK${p.amountHKD}</td>
                      <td className="p-3 text-right">{p.creditsAmount}</td>
                      <td className="p-3 text-right">
                        <Badge
                          variant={
                            p.status === "completed"
                              ? "default"
                              : p.status === "pending"
                                ? "secondary"
                                : "destructive"
                          }
                        >
                          {p.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <div className="text-center mt-8">
        <Link href="/discuss">
          <Button variant="ghost">
            <MessageSquareText className="mr-2 h-4 w-4" />
            Back to Discuss
          </Button>
        </Link>
      </div>
    </div>
  );
}
