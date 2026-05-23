"use client";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { TranscriptDisplay } from "@/components/discuss/transcript-display";
import { LearningNotes } from "@/components/discuss/learning-notes";
import { AudioPlayer } from "@/components/discuss/audio-player";
import type {
  DialogueItem,
  LearningNotes as LearningNotesType,
} from "@/types";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  Loader2,
  Coins,
  ShieldCheck,
  ShoppingCart,
  LogIn,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Pagination } from "@/components/admin-pagination";

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

interface SignInRow {
  id: string;
  userName: string | null;
  email: string | null;
  provider: string;
  createdAt: string;
}

type DiscussionSortKey = "userName" | "createdAt" | "title" | "dialogueMode";
type PurchaseSortKey = "userName" | "createdAt" | "planName" | "amountHKD";
type SignInSortKey = "userName" | "createdAt";

interface DetailSession {
  id: string;
  title: string;
  dialogueMode: string;
  inputText: string | null;
  transcript: DialogueItem[];
  learningNotes: LearningNotesType;
  audioUrl: string | null;
  audioExpiresAt: string | null;
  accessCode: string | null;
  ttsCostHKD: number;
  usedOwnApiKey: boolean;
  generationCost: number;
  createdAt: string;
}

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
  const [signIns, setSignIns] = useState<SignInRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [generationCost, setGenerationCost] = useState(10);
  const [search, setSearch] = useState("");
  const [dSortBy, setDSortBy] = useState<DiscussionSortKey>("createdAt");
  const [dSortDesc, setDSortDesc] = useState(true);
  const [pSortBy, setPSortBy] = useState<PurchaseSortKey>("createdAt");
  const [pSortDesc, setPSortDesc] = useState(true);
  const [sSortBy, setSSortBy] = useState<SignInSortKey>("createdAt");
  const [sSortDesc, setSSortDesc] = useState(true);
  const [dPage, setDPage] = useState(1);
  const [dPageSize, setDPageSize] = useState(20);
  const [pPage, setPPage] = useState(1);
  const [pPageSize, setPPagesize] = useState(20);
  const [sPage, setSPage] = useState(1);
  const [sPageSize, setSPagesize] = useState(20);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [detailSession, setDetailSession] = useState<DetailSession | null>(
    null
  );
  const [detailLoading, setDetailLoading] = useState(false);
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
      setDPage(1);
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
      setPPage(1);
    },
    [pSortBy]
  );

  const toggleSSort = useCallback(
    (key: string) => {
      const k = key as SignInSortKey;
      if (sSortBy === k) {
        setSSortDesc((prev) => !prev);
      } else {
        setSSortBy(k);
        setSSortDesc(false);
      }
      setSPage(1);
    },
    [sSortBy]
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

    async function loadSignIns() {
      try {
        const params = new URLSearchParams({
          sortBy: sSortBy,
          sortOrder: sSortDesc ? "desc" : "asc",
        });
        const res = await fetch(`/api/admin/sign-ins?${params}`);
        if (res.ok && !cancelled) {
          const data = await res.json();
          setSignIns(data.signIns || []);
        }
      } catch {}
    }

    if (initialLoad.current) {
      Promise.all([loadDiscussions(), loadPurchases(), loadSignIns()]).finally(
        () => {
          if (!cancelled) {
            setLoading(false);
            initialLoad.current = false;
          }
        }
      );
    } else {
      loadDiscussions();
      loadPurchases();
      loadSignIns();
    }

    return () => {
      cancelled = true;
    };
  }, [dSortBy, dSortDesc, pSortBy, pSortDesc, sSortBy, sSortDesc, search]);

  useEffect(() => {
    if (!detailId) {
      setDetailSession(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    async function loadDetail() {
      try {
        const res = await fetch(`/api/admin/discussions/${detailId}`);
        if (res.ok && !cancelled) {
          const data = await res.json();
          setDetailSession(data);
        }
      } catch {
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    }
    loadDetail();
    return () => {
      cancelled = true;
    };
  }, [detailId]);

  const userCumulative = useMemo(() => {
    const map = new Map<string, { credits: number; ttsHKD: number }>();
    for (const d of discussions) {
      const key = d.email || d.userName || "";
      const prev = map.get(key) || { credits: 0, ttsHKD: 0 };
      if (!d.usedOwnApiKey) prev.credits += generationCost;
      prev.ttsHKD += d.ttsCostHKD;
      map.set(key, prev);
    }
    return map;
  }, [discussions, generationCost]);

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
          <TabsTrigger value="signins" className="gap-1.5">
            <LogIn className="h-4 w-4" />
            Sign-ins
            <Badge variant="secondary" className="ml-1 text-xs">
              {signIns.length}
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
              onChange={(e) => {
                setSearch(e.target.value);
                setDPage(1);
              }}
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
            <>
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
                      <th className="p-3 text-left font-medium">
                        Cumulative
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {discussions
                      .slice(
                        (Math.min(dPage, Math.max(1, Math.ceil(discussions.length / dPageSize))) - 1) * dPageSize,
                        Math.min(dPage, Math.max(1, Math.ceil(discussions.length / dPageSize))) * dPageSize
                      )
                      .map((d) => (
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
                          <td className="p-3 max-w-[300px]">
                            <button
                              type="button"
                              className="truncate text-left hover:underline cursor-pointer"
                              title={d.title}
                              onClick={() => setDetailId(d.id)}
                            >
                              {d.title}
                            </button>
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
                                <span className="text-muted-foreground">
                                  Own key
                                </span>
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
                          <td className="p-3 whitespace-nowrap text-xs text-muted-foreground">
                            {(() => {
                              const cum = userCumulative.get(d.email || d.userName || "");
                              if (!cum) return null;
                              const parts: string[] = [];
                              if (cum.credits > 0) parts.push(`${cum.credits} credits`);
                              if (cum.ttsHKD > 0) parts.push(`HK${cum.ttsHKD.toFixed(2)}`);
                              return parts.length > 0 ? parts.join(" + ") : "—";
                            })()}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                total={discussions.length}
                pageSize={dPageSize}
                currentPage={dPage}
                onPageChange={setDPage}
                onPageSizeChange={(size) => {
                  setDPageSize(size);
                  setDPage(1);
                }}
              />
            </>
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
            <>
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
                    {purchases
                      .slice(
                        (Math.min(pPage, Math.max(1, Math.ceil(purchases.length / pPageSize))) - 1) * pPageSize,
                        Math.min(pPage, Math.max(1, Math.ceil(purchases.length / pPageSize))) * pPageSize
                      )
                      .map((p) => (
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
              <Pagination
                total={purchases.length}
                pageSize={pPageSize}
                currentPage={pPage}
                onPageChange={setPPage}
                onPageSizeChange={(size) => {
                  setPPagesize(size);
                  setPPage(1);
                }}
              />
            </>
          )}
        </TabsContent>

        <TabsContent value="signins" className="mt-4 space-y-4">
          {signIns.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">
                  No sign-in records found.
                </p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="rounded-md border overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <SortableTh
                        label="User"
                        sortKey="userName"
                        activeSortKey={sSortBy}
                        isDesc={sSortDesc}
                        onSort={toggleSSort}
                      />
                      <SortableTh
                        label="Date &amp; Time"
                        sortKey="createdAt"
                        activeSortKey={sSortBy}
                        isDesc={sSortDesc}
                        onSort={toggleSSort}
                      />
                      <th className="p-3 text-left font-medium">Provider</th>
                    </tr>
                  </thead>
                  <tbody>
                    {signIns
                      .slice(
                        (Math.min(sPage, Math.max(1, Math.ceil(signIns.length / sPageSize))) - 1) * sPageSize,
                        Math.min(sPage, Math.max(1, Math.ceil(signIns.length / sPageSize))) * sPageSize
                      )
                      .map((s) => (
                        <tr
                          key={s.id}
                          className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                        >
                          <td className="p-3">
                            <div className="font-medium">
                              {s.userName || "Unknown"}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {s.email}
                            </div>
                          </td>
                          <td className="p-3 whitespace-nowrap">
                            {formatDateHK(s.createdAt)}
                          </td>
                          <td className="p-3">
                            <Badge variant="secondary">{s.provider}</Badge>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                total={signIns.length}
                pageSize={sPageSize}
                currentPage={sPage}
                onPageChange={setSPage}
                onPageSizeChange={(size) => {
                  setSPagesize(size);
                  setSPage(1);
                }}
              />
            </>
          )}
        </TabsContent>
      </Tabs>

      <Dialog
        open={!!detailId}
        onOpenChange={(open) => {
          if (!open) setDetailId(null);
        }}
      >
        <DialogContent className="sm:max-w-3xl md:max-w-4xl lg:max-w-5xl max-h-[90vh] overflow-y-auto">
          {detailLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">Loading...</span>
            </div>
          ) : detailSession ? (
            <div className="space-y-4">
              <DialogHeader>
                <DialogTitle>{detailSession.title}</DialogTitle>
              </DialogHeader>
              <p className="text-sm text-muted-foreground">
                {new Date(detailSession.createdAt).toLocaleString("en-HK", {
                  timeZone: "Asia/Hong_Kong",
                })}
                {" | "}
                {detailSession.usedOwnApiKey
                  ? `HK$${detailSession.ttsCostHKD.toFixed(2)}`
                  : `${detailSession.generationCost} credits`}
              </p>
              <Separator />
              {detailSession.inputText && (
                <Card>
                  <CardHeader>
                    <CardTitle>📌 Task 任務</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-64 overflow-y-auto rounded-md bg-muted p-4">
                      <pre className="whitespace-pre-wrap text-sm leading-relaxed font-sans">
                        {detailSession.inputText}
                      </pre>
                    </div>
                  </CardContent>
                </Card>
              )}
              {detailSession.audioUrl &&
              detailSession.audioExpiresAt &&
              new Date(detailSession.audioExpiresAt).getTime() > Date.now() ? (
                <AudioPlayer
                  src={detailSession.audioUrl}
                  expiryDays={
                    (new Date(detailSession.audioExpiresAt).getTime() -
                      Date.now()) /
                    (24 * 60 * 60 * 1000)
                  }
                  accessCode={detailSession.accessCode}
                />
              ) : detailSession.audioUrl || detailSession.accessCode ? (
                <Card>
                  <CardHeader>
                    <CardTitle>🎧 Audio 錄音</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Audio has expired and is no longer available.
                    </p>
                    {detailSession.accessCode && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Access code: {detailSession.accessCode}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ) : null}
              <TranscriptDisplay items={detailSession.transcript} />
              <LearningNotes notes={detailSession.learningNotes} />
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
