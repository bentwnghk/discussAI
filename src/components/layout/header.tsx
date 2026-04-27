"use client";

import { useState } from "react";
import Link from "next/link";
import { useSession, signOut } from "next-auth/react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { SettingsDialog } from "@/components/settings-dialog";
import { Settings, LogOut, MessageSquareText, History, LogIn, Coins } from "lucide-react";
import { useCredits } from "@/hooks/use-credits";

export function Header() {
  const { data: session } = useSession();
  const { balance } = useCredits();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-bold font-heading">Mr.🆖 DiscussAI</span>
          <span className="hidden sm:inline text-sm text-muted-foreground">
            HKDSE Oral Practice
          </span>
        </Link>

        {session?.user ? (
          <div className="flex items-center gap-4">
            <nav className="hidden md:flex items-center gap-4">
              <Link href="/discuss">
                <Button variant="ghost" size="sm" className="font-heading">
                  <MessageSquareText className="mr-2 h-4 w-4" />
                  Discuss
                </Button>
              </Link>
              <Link href="/history">
                <Button variant="ghost" size="sm" className="font-heading">
                  <History className="mr-2 h-4 w-4" />
                  History
                </Button>
              </Link>
              <Link href="/credits">
                <Button variant="ghost" size="sm" className="font-heading">
                  <Coins className="mr-2 h-4 w-4" />
                  {balance !== null ? `${balance} Credits` : "Credits"}
                </Button>
              </Link>
            </nav>
            <DropdownMenu>
              <DropdownMenuTrigger
                className="relative flex h-8 w-8 items-center justify-center rounded-full hover:bg-accent"
              >
                <Avatar className="h-8 w-8">
                  <AvatarImage
                    src={session.user.image || ""}
                    alt={session.user.name || ""}
                  />
                  <AvatarFallback>
                    {session.user.name?.[0]?.toUpperCase() || "U"}
                  </AvatarFallback>
                </Avatar>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-[16rem]">
                <div className="px-2 py-1.5 text-sm font-medium">
                  {session.user.name}
                </div>
                <div className="px-2 py-1.5 text-xs text-muted-foreground">
                  {session.user.email}
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="md:hidden font-heading">
                  <Link href="/discuss" className="flex items-center w-full">
                    <MessageSquareText className="mr-2 h-4 w-4" />
                    Discuss
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem className="md:hidden font-heading">
                  <Link href="/history" className="flex items-center w-full">
                    <History className="mr-2 h-4 w-4" />
                    History
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem className="md:hidden font-heading">
                  <Link href="/credits" className="flex items-center w-full">
                    <Coins className="mr-2 h-4 w-4" />
                    {balance !== null ? `${balance} Credits` : "Credits"}
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator className="md:hidden" />
                <DropdownMenuItem className="font-heading" onClick={() => setSettingsOpen(true)}>
                  <Settings className="mr-2 h-4 w-4" />
                  Settings
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="font-heading"
                  onClick={() => signOut({ callbackUrl: "/" })}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <SettingsDialog
              open={settingsOpen}
              onOpenChange={setSettingsOpen}
            />
          </div>
        ) : (
          <Link href="/login">
            <Button size="sm">
              <LogIn className="mr-2 h-4 w-4" />
              Sign in
            </Button>
          </Link>
        )}
      </div>
    </header>
  );
}
