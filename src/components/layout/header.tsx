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

export function Header() {
  const { data: session } = useSession();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-bold">Mr.🆖 DiscussAI</span>
          <span className="hidden sm:inline text-sm text-muted-foreground">
            HKDSE Oral Practice
          </span>
        </Link>

        {session?.user ? (
          <div className="flex items-center gap-4">
            <nav className="hidden md:flex items-center gap-4">
              <Link href="/discuss">
                <Button variant="ghost" size="sm">
                  Discuss
                </Button>
              </Link>
              <Link href="/history">
                <Button variant="ghost" size="sm">
                  History
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
                <DropdownMenuItem className="md:hidden">
                  <Link href="/discuss" className="w-full">
                    Discuss
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem className="md:hidden">
                  <Link href="/history" className="w-full">
                    History
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator className="md:hidden" />
                <DropdownMenuItem onClick={() => setSettingsOpen(true)}>
                  Settings
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => signOut({ callbackUrl: "/" })}
                >
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
            <Button size="sm">Sign in</Button>
          </Link>
        )}
      </div>
    </header>
  );
}
