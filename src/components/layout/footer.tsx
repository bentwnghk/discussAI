import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t py-6 text-center text-sm text-muted-foreground">
      <p>
        Built with ❤️ by Mr.🆖 for students learning English. Powered by{" "}
        <Link
          href="https://api.mr5ai.com"
          className="underline underline-offset-4 hover:text-foreground transition-colors"
          target="_blank"
          rel="noopener noreferrer"
        >
          Mr. AI Hub
        </Link>
      </p>
    </footer>
  );
}
