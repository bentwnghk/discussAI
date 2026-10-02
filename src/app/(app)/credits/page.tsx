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
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Coins, CheckCircle, XCircle, Loader2, Star, ShoppingCart, KeyRound, Zap, Package, Users, User, Activity } from "lucide-react";
import { SettingsDialog } from "@/components/settings-dialog";

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

interface UsageRecord {
  id: string;
  amount: number;
  type: string;
  description: string | null;
  createdAt: string;
}

export default function CreditsPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { balance, refreshBalance } = useCredits();
  const [plans, setPlans] = useState<PlanConfig[]>([]);
  const [generationCost, setGenerationCost] = useState(10);
  const [responseCost, setResponseCost] = useState(2);
  const [loading, setLoading] = useState<string | null>(null);
  const [purchases, setPurchases] = useState<PurchaseRecord[]>([]);
  const [transactions, setTransactions] = useState<UsageRecord[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
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
        if (data.responseCost) setResponseCost(data.responseCost);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/user/purchases")
      .then((res) => res.json())
      .then((data) => setPurchases(data.purchases || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/user/transactions")
      .then((res) => res.json())
      .then((data) => setTransactions(data.transactions || []))
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
          Purchase credits to generate discussions and individual responses
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
          const responses = Math.floor(plan.credits / responseCost);
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
                <p className="text-sm text-muted-foreground space-y-1">
                  <span className="flex items-center justify-center gap-1.5">
                    <Users className="h-3.5 w-3.5" />
                    {discussions} group discussions
                  </span>
                  <span className="flex items-center justify-center text-xs font-medium uppercase tracking-wider">
                    or
                  </span>
                  <span className="flex items-center justify-center gap-1.5">
                    <User className="h-3.5 w-3.5" />
                    {responses} individual responses
                  </span>
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

      {(transactions.length > 0 || purchases.length > 0) && (
        <>
          <Separator className="my-8" />
          <div>
            <h2 className="text-xl font-semibold mb-4">History</h2>
            <Tabs defaultValue="usage">
              <TabsList>
                <TabsTrigger value="usage" className="gap-1.5">
                  <Activity className="h-4 w-4" />
                  Usage History
                </TabsTrigger>
                <TabsTrigger value="purchases" className="gap-1.5">
                  <ShoppingCart className="h-4 w-4" />
                  Purchase History
                </TabsTrigger>
              </TabsList>

              <TabsContent value="usage" className="mt-4">
                {transactions.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-6 text-center">
                    No credit usage yet.
                  </p>
                ) : (
                  <div className="rounded-md border max-h-[400px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 z-10 bg-muted">
                        <tr className="border-b">
                          <th className="p-3 text-left font-medium">Date &amp; Time</th>
                          <th className="p-3 text-left font-medium">Type</th>
                          <th className="p-3 text-left font-medium">Details</th>
                          <th className="p-3 text-right font-medium">Credits</th>
                        </tr>
                      </thead>
                      <tbody>
                        {transactions.map((t) => {
                          const isRefund = t.type === "refund";
                          const details = t.description?.startsWith("Refund for ")
                            ? "Refund for failed generation"
                            : t.description || "Credit usage";
                          return (
                            <tr key={t.id} className="border-b last:border-0">
                              <td className="p-3 whitespace-nowrap">
                                {new Date(t.createdAt).toLocaleString("en-HK", {
                                  timeZone: "Asia/Hong_Kong",
                                  year: "numeric",
                                  month: "short",
                                  day: "numeric",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </td>
                              <td className="p-3">
                                <Badge variant={isRefund ? "secondary" : "outline"}>
                                  {isRefund ? "Refund" : "Usage"}
                                </Badge>
                              </td>
                              <td className="p-3">{details}</td>
                              <td
                                className={`p-3 text-right font-medium whitespace-nowrap ${
                                  t.amount < 0
                                    ? "text-red-600 dark:text-red-400"
                                    : "text-green-600 dark:text-green-400"
                                }`}
                              >
                                {t.amount > 0 ? `+${t.amount}` : t.amount}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="purchases" className="mt-4">
                {purchases.length === 0 ? (
                  <p className="text-sm text-muted-foreground py-6 text-center">
                    No purchases yet.
                  </p>
                ) : (
                  <div className="rounded-md border max-h-[400px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 z-10 bg-muted">
                        <tr className="border-b">
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
                )}
              </TabsContent>
            </Tabs>
          </div>
        </>
      )}

      <div className="text-center mt-8">
        <Button variant="ghost" onClick={() => setSettingsOpen(true)}>
          <KeyRound className="mr-2 h-4 w-4" />
          Already have your own API key?
        </Button>
      </div>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </div>
  );
}
