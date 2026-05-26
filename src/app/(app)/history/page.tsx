"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Plus,
  Eye,
  Pencil,
  Trash2,
  X,
  Check,
  Coins,
  MessageSquareText,
  Mic,
  History,
  ChevronLeft,
  ChevronRight,
  Search,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { SessionType } from "@/types";

interface HistoryItem {
  id: string;
  title: string;
  sessionType: SessionType;
  dialogueMode: string;
  inputMethod: string;
  charactersCount: number;
  ttsCostHKD: number;
  usedOwnApiKey: boolean;
  createdAt: string;
}

type FilterType = "all" | "discussion" | "response";

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [generationCost, setGenerationCost] = useState(10);
  const [responseCost, setResponseCost] = useState(2);
  const [loading, setLoading] = useState(true);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameTitle, setRenameTitle] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [pageSize, setPageSize] = useState(10);
  const [currentPage, setCurrentPage] = useState(1);
  const [filterType, setFilterType] = useState<FilterType>("all");

  const filteredItems = items.filter((item) => {
    if (filterType !== "all" && item.sessionType !== filterType) return false;
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      item.title.toLowerCase().includes(q) ||
      item.dialogueMode.toLowerCase().includes(q) ||
      item.inputMethod.toLowerCase().includes(q) ||
      item.sessionType.toLowerCase().includes(q) ||
      new Date(item.createdAt)
        .toLocaleString("en-HK", { timeZone: "Asia/Hong_Kong" })
        .toLowerCase()
        .includes(q)
    );
  });

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / pageSize));
  const safePage = Math.min(currentPage, totalPages);
  const paginatedItems = filteredItems.slice(
    (safePage - 1) * pageSize,
    safePage * pageSize
  );

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/history");
        if (res.ok && !cancelled) {
          const data = await res.json();
          setItems(data.sessions || data);
          if (data.generationCost) setGenerationCost(data.generationCost);
          if (data.responseCost) setResponseCost(data.responseCost);
        }
      } catch {
        // silently fail
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const handleDelete = useCallback(async () => {
    if (!deleteId) return;
    try {
      await fetch(`/api/history/${deleteId}`, { method: "DELETE" });
      setItems((prev) => prev.filter((i) => i.id !== deleteId));
    } catch {
      // silently fail
    }
    setDeleteId(null);
  }, [deleteId]);

  const handleRename = useCallback(async () => {
    if (!renameId || !renameTitle.trim()) return;
    try {
      const res = await fetch(`/api/history/${renameId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: renameTitle.trim() }),
      });
      if (res.ok) {
        setItems((prev) =>
          prev.map((i) =>
            i.id === renameId ? { ...i, title: renameTitle.trim() } : i
          )
        );
      }
    } catch {
      // silently fail
    }
    setRenameId(null);
    setRenameTitle("");
  }, [renameId, renameTitle]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <p className="text-center text-muted-foreground">Loading history...</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-6">
        <div className="flex items-center gap-2">
          <History className="size-6" />
          <h1 className="text-2xl font-bold">Practice History</h1>
          <Badge variant="secondary" className="gap-1">
            <MessageSquareText className="h-3 w-3" />
            {items.filter((i) => i.sessionType !== "response").length}
          </Badge>
          <Badge variant="secondary" className="gap-1">
            <Mic className="h-3 w-3" />
            {items.filter((i) => i.sessionType === "response").length}
          </Badge>
        </div>
        <div className="flex gap-2">
          <Link href="/discuss">
            <Button variant="outline" size="sm">
              <Plus className="mr-2 h-4 w-4" />
              Discussion
            </Button>
          </Link>
          <Link href="/respond">
            <Button variant="outline" size="sm">
              <Plus className="mr-2 h-4 w-4" />
              Response
            </Button>
          </Link>
        </div>
      </div>

      <div className="flex gap-2 mb-4">
        <Button
          variant={filterType === "all" ? "default" : "outline"}
          size="sm"
          onClick={() => { setFilterType("all"); setCurrentPage(1); }}
        >
          All
        </Button>
        <Button
          variant={filterType === "discussion" ? "default" : "outline"}
          size="sm"
          onClick={() => { setFilterType("discussion"); setCurrentPage(1); }}
          className="gap-1"
        >
          <MessageSquareText className="h-3 w-3" />
          Discussions
        </Button>
        <Button
          variant={filterType === "response" ? "default" : "outline"}
          size="sm"
          onClick={() => { setFilterType("response"); setCurrentPage(1); }}
          className="gap-1"
        >
          <Mic className="h-3 w-3" />
          Responses
        </Button>
      </div>

      {items.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">No practice sessions yet.</p>
            <div className="flex justify-center gap-3 mt-4">
              <Link href="/discuss">
                <Button>
                  <MessageSquareText className="mr-2 h-4 w-4" />
                  Start a Discussion
                </Button>
              </Link>
              <Link href="/respond">
                <Button variant="outline">
                  <Mic className="mr-2 h-4 w-4" />
                  Start a Response
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search by title, mode, method, or date..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              className="pl-9"
            />
          </div>

          {filteredItems.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">
                  No sessions match &quot;{searchQuery}&quot;
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-3"
                  onClick={() => setSearchQuery("")}
                >
                  Clear search
                </Button>
              </CardContent>
            </Card>
          ) : (
            <>
              {paginatedItems.map((item) => (
                <Card key={item.id}>
                  <CardContent className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-4 gap-2">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{item.title}</p>
                        {item.sessionType === "response" ? (
                          <Badge variant="secondary" className="gap-1 text-xs">
                            <Mic className="h-3 w-3" />
                            Response
                          </Badge>
                        ) : (
                          <Badge variant="secondary" className="gap-1 text-xs">
                            <MessageSquareText className="h-3 w-3" />
                            Discussion
                          </Badge>
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <span>
                          {new Date(item.createdAt).toLocaleString("en-HK", {
                            timeZone: "Asia/Hong_Kong",
                          })}
                        </span>
                        <Badge variant="secondary">{item.dialogueMode}</Badge>
                        {!item.usedOwnApiKey && (
                          <Badge variant="outline" className="gap-1">
                            <Coins className="h-3 w-3" />
                            {item.sessionType === "response" ? responseCost : generationCost}
                          </Badge>
                        )}
                        {item.usedOwnApiKey && (
                          <span>HK${item.ttsCostHKD.toFixed(2)}</span>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Link href={`/history/${item.id}`}>
                        <Button variant="outline" size="sm">
                          <Eye className="mr-2 h-4 w-4" />
                          Load
                        </Button>
                      </Link>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setRenameId(item.id);
                          setRenameTitle(item.title);
                        }}
                      >
                        <Pencil className="mr-2 h-4 w-4" />
                        Rename
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => setDeleteId(item.id)}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}

              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <span>Per page</span>
                    <Select
                      value={String(pageSize)}
                      onValueChange={(v) => {
                        setPageSize(Number(v));
                        setCurrentPage(1);
                      }}
                    >
                      <SelectTrigger size="sm" className="w-[72px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="10">10</SelectItem>
                        <SelectItem value="20">20</SelectItem>
                        <SelectItem value="50">50</SelectItem>
                      </SelectContent>
                    </Select>
                    <span>
                      {(safePage - 1) * pageSize + 1}–
                      {Math.min(safePage * pageSize, filteredItems.length)} of{" "}
                      {filteredItems.length}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={safePage <= 1}
                      onClick={() => setCurrentPage(safePage - 1)}
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <span className="px-2 text-sm text-muted-foreground">
                      {safePage} / {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={safePage >= totalPages}
                      onClick={() => setCurrentPage(safePage + 1)}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      <Dialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Session</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Are you sure you want to delete this session? This action cannot
            be undone.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteId(null)}>
              <X className="mr-2 h-4 w-4" />
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete}>
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={!!renameId} onOpenChange={() => setRenameId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Session</DialogTitle>
          </DialogHeader>
          <Input
            value={renameTitle}
            onChange={(e) => setRenameTitle(e.target.value)}
            placeholder="Enter new title"
          />
          <DialogFooter>
            <Button onClick={handleRename}>
              <Check className="mr-2 h-4 w-4" />
              Rename
            </Button>
            <Button variant="outline" onClick={() => setRenameId(null)}>
              <X className="mr-2 h-4 w-4" />
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
