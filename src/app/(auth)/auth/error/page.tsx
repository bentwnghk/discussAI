import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function AuthErrorPage() {
  return (
    <div className="container mx-auto flex min-h-[80vh] items-center justify-center px-4">
      <div className="mx-auto max-w-sm space-y-6 text-center">
        <h1 className="text-2xl font-bold">Authentication Error</h1>
        <p className="text-muted-foreground">
          There was a problem signing you in. Please try again.
        </p>
        <Link href="/login">
          <Button>Try Again</Button>
        </Link>
      </div>
    </div>
  );
}
