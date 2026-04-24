import { SignIn } from "@/components/sign-in";

export default function LoginPage() {
  return (
    <div className="container mx-auto flex min-h-[80vh] items-center justify-center px-4">
      <div className="mx-auto max-w-sm space-y-6 text-center">
        <div>
          <h1 className="text-2xl font-bold">Sign in to DiscussAI</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Sign in with your Google account to start practicing
          </p>
        </div>
        <SignIn />
      </div>
    </div>
  );
}
