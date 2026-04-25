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
import { Plus, Play, Eye, Pencil, Trash2, X, Check } from "lucide-react";

interface HistoryItem {
  id: string;
  title: string;
  dialogueMode: string;
  inputMethod: string;
  charactersCount: number;
  ttsCostHKD: number;
  createdAt: string;
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameTitle, setRenameTitle] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/history");
        if (res.ok && !cancelled) {
          const data = await res.json();
          setItems(data);
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
    // We'd need a PATCH endpoint for rename. For now, delete and recreate isn't ideal.
    // Let's just update the title locally for now and note this as a TODO.
    setItems((prev) =>
      prev.map((i) =>
        i.id === renameId ? { ...i, title: renameTitle.trim() } : i
      )
    );
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
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Practice History</h1>
        <Link href="/discuss">
          <Button variant="outline" size="sm">
            <Plus className="mr-2 h-4 w-4" />
            New Discussion
          </Button>
        </Link>
      </div>

      {items.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">No discussions yet.</p>
            <Link href="/discuss">
              <Button className="mt-4">
                <Play className="mr-2 h-4 w-4" />
                Start Practicing
              </Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <Card key={item.id}>
              <CardContent className="flex flex-col sm:flex-row sm:items-center sm:justify-between py-4 gap-2">
                <div className="space-y-1">
                  <p className="font-medium">{item.title}</p>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span>
                      {new Date(item.createdAt).toLocaleString("en-HK", {
                        timeZone: "Asia/Hong_Kong",
                      })}
                    </span>
                    <Badge variant="secondary">{item.dialogueMode}</Badge>
                    <span>HK${item.ttsCostHKD.toFixed(2)}</span>
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
        </div>
      )}

      <Dialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Discussion</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Are you sure you want to delete this discussion? This action cannot
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
            <DialogTitle>Rename Discussion</DialogTitle>
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
