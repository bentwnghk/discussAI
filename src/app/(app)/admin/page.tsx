"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  Loader2,
  Coins,
  ShieldCheck,
  ShoppingCart,
} from "lucide-react";

interface DiscussionRow {
  id: string;
  userName: string | null;
  email: string | null;
  title: string;
  dialogueMode: string;
  createdAt: string;
  usedOwnApiKey: boolean;
  ttsCostHKD: number;
}

interface PurchaseRow {
  id: string;
  userName: string | null;
  email: string | null;
  planName: string;
  creditsAmount: number;
  amountHKD: number;
  status: string;
  createdAt: string;
}

type DiscussionSortKey = "userName" | "createdAt" | "title" | "dialogueMode";
type PurchaseSortKey = "userName" | "createdAt" | "planName" | "amountHKD";

function formatDateHK(dateStr: string): string {
  return new Date(dateStr).toLocaleString("en-HK", {
    timeZone: "Asia/Hong_Kong",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SortIcon({ active, desc }: { active: boolean; desc: boolean }) {
  if (!active) return <ArrowUpDown className="ml-1 h-3 w-3 opacity-50" />;
  return desc ? (
    <ArrowDown className="ml-1 h-3 w-3" />
  ) : (
    <ArrowUp className="ml-1 h-3 w-3" />
  );
}

function SortableTh({
  label,
  sortKey,
  activeSortKey,
  isDesc,
  onSort,
}: {
  label: string;
  sortKey: string;
  activeSortKey: string;
  isDesc: boolean;
  onSort: (key: string) => void;
}) {
  return (
    <th
      className="p-3 text-left font-medium cursor-pointer select-none hover:bg-muted/80 transition-colors"
      onClick={() => onSort(sortKey)}
    >
      <span className="inline-flex items-center">
        {label}
        <SortIcon active={activeSortKey === sortKey} desc={isDesc} />
      </span>
    </th>
  );
}

export default function AdminDashboardPage() {
  const [discussions, setDiscussions] = useState<DiscussionRow[]>([]);
  const [purchases, setPurchases] = useState<PurchaseRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [generationCost, setGenerationCost] = useState(10);
  const [search, setSearch] = useState("");
  const [dSortBy, setDSortBy] = useState<DiscussionSortKey>("createdAt");
  const [dSortDesc, setDSortDesc] = useState(true);
  const [pSortBy, setPSortBy] = useState<PurchaseSortKey>("createdAt");
  const [pSortDesc, setPSortDesc] = useState(true);
  const initialLoad = useRef(true);

  const toggleDSort = useCallback(
    (key: string) => {
      const k = key as DiscussionSortKey;
      if (dSortBy === k) {
        setDSortDesc((prev) => !prev);
      } else {
        setDSortBy(k);
        setDSortDesc(false);
      }
    },
    [dSortBy]
  );

  const togglePSort = useCallback(
    (key: string) => {
      const k = key as PurchaseSortKey;
      if (pSortBy === k) {
        setPSortDesc((prev) => !prev);
      } else {
        setPSortBy(k);
        setPSortDesc(false);
      }
    },
    [pSortBy]
  );

  useEffect(() => {
    let cancelled = false;
    async function loadDiscussions() {
      try {
        const params = new URLSearchParams({
          sortBy: dSortBy,
          sortOrder: dSortDesc ? "desc" : "asc",
          q: search,
        });
        const res = await fetch(`/api/admin/discussions?${params}`);
        if (res.ok && !cancelled) {
          const data = await res.json();
          setDiscussions(data.discussions || []);
          if (data.generationCost) setGenerationCost(data.generationCost);
        }
      } catch {}
    }

    async function loadPurchases() {
      try {
        const params = new URLSearchParams({
          sortBy: pSortBy,
          sortOrder: pSortDesc ? "desc" : "asc",
        });
        const res = await fetch(`/api/admin/purchases?${params}`);
        if (res.ok && !cancelled) {
          const data = await res.json();
          setPurchases(data.purchases || []);
        }
      } catch {}
    }

    if (initialLoad.current) {
      Promise.all([loadDiscussions(), loadPurchases()]).finally(() => {
        if (!cancelled) {
          setLoading(false);
          initialLoad.current = false;
        }
      });
    } else {
      loadDiscussions();
      loadPurchases();
    }

    return () => {
      cancelled = true;
    };
  }, [dSortBy, dSortDesc, pSortBy, pSortDesc, search]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">
            Loading dashboard...
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex items-center gap-3 mb-6">
        <ShieldCheck className="size-6" />
        <h1 className="text-2xl font-bold">Admin Dashboard</h1>
      </div>

      <Tabs defaultValue="usage">
        <TabsList>
          <TabsTrigger value="usage" className="gap-1.5">
            <Coins className="h-4 w-4" />
            Usage
            <Badge variant="secondary" className="ml-1 text-xs">
              {discussions.length}
            </Badge>
          </TabsTrigger>
          <TabsTrigger value="purchases" className="gap-1.5">
            <ShoppingCart className="h-4 w-4" />
            Purchases
            <Badge variant="secondary" className="ml-1 text-xs">
              {purchases.length}
            </Badge>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="usage" className="mt-4 space-y-4">
          <div className="relative max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search by name, email, title, or mode..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>

          {discussions.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">
                  {search
                    ? `No discussions match "${search}"`
                    : "No discussion sessions found."}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="rounded-md border overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <SortableTh
                      label="User"
                      sortKey="userName"
                      activeSortKey={dSortBy}
                      isDesc={dSortDesc}
                      onSort={toggleDSort}
                    />
                    <SortableTh
                      label="Date"
                      sortKey="createdAt"
                      activeSortKey={dSortBy}
                      isDesc={dSortDesc}
                      onSort={toggleDSort}
                    />
                    <SortableTh
                      label="Title"
                      sortKey="title"
                      activeSortKey={dSortBy}
                      isDesc={dSortDesc}
                      onSort={toggleDSort}
                    />
                    <SortableTh
                      label="Mode"
                      sortKey="dialogueMode"
                      activeSortKey={dSortBy}
                      isDesc={dSortDesc}
                      onSort={toggleDSort}
                    />
                    <th className="p-3 text-left font-medium">
                      Credits / Cost
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {discussions.map((d) => (
                    <tr
                      key={d.id}
                      className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                    >
                      <td className="p-3">
                        <div className="font-medium">
                          {d.userName || "Unknown"}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {d.email}
                        </div>
                      </td>
                      <td className="p-3 whitespace-nowrap">
                        {formatDateHK(d.createdAt)}
                      </td>
                      <td className="p-3 max-w-[300px] truncate">
                        {d.title}
                      </td>
                      <td className="p-3">
                        <Badge
                          variant={
                            d.dialogueMode === "Deeper"
                              ? "default"
                              : "secondary"
                          }
                        >
                          {d.dialogueMode}
                        </Badge>
                      </td>
                      <td className="p-3 whitespace-nowrap">
                        <div>
                          {d.usedOwnApiKey ? (
                            <span className="text-muted-foreground">Own key</span>
                          ) : (
                            <Badge variant="outline" className="gap-1">
                              <Coins className="h-3 w-3" />
                              {generationCost} credits
                            </Badge>
                          )}
                        </div>
                        {d.ttsCostHKD > 0 && (
                          <div className="text-xs text-muted-foreground mt-0.5">
                            TTS HK${d.ttsCostHKD.toFixed(2)}
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </TabsContent>

        <TabsContent value="purchases" className="mt-4 space-y-4">
          {purchases.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">No purchases found.</p>
              </CardContent>
            </Card>
          ) : (
            <div className="rounded-md border overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <SortableTh
                      label="User"
                      sortKey="userName"
                      activeSortKey={pSortBy}
                      isDesc={pSortDesc}
                      onSort={togglePSort}
                    />
                    <SortableTh
                      label="Date"
                      sortKey="createdAt"
                      activeSortKey={pSortBy}
                      isDesc={pSortDesc}
                      onSort={togglePSort}
                    />
                    <SortableTh
                      label="Package"
                      sortKey="planName"
                      activeSortKey={pSortBy}
                      isDesc={pSortDesc}
                      onSort={togglePSort}
                    />
                    <SortableTh
                      label="Amount Paid"
                      sortKey="amountHKD"
                      activeSortKey={pSortBy}
                      isDesc={pSortDesc}
                      onSort={togglePSort}
                    />
                    <th className="p-3 text-left font-medium">Credits</th>
                    <th className="p-3 text-left font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {purchases.map((p) => (
                    <tr
                      key={p.id}
                      className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                    >
                      <td className="p-3">
                        <div className="font-medium">
                          {p.userName || "Unknown"}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {p.email}
                        </div>
                      </td>
                      <td className="p-3 whitespace-nowrap">
                        {formatDateHK(p.createdAt)}
                      </td>
                      <td className="p-3">{p.planName}</td>
                      <td className="p-3 whitespace-nowrap font-medium">
                        HK${p.amountHKD.toFixed(2)}
                      </td>
                      <td className="p-3">{p.creditsAmount}</td>
                      <td className="p-3">
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
  );
}
