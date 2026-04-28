import { auth } from "@/lib/auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  const { pathname } = req.nextUrl;
  const isAuthPage = pathname.startsWith("/login") || pathname.startsWith("/auth");
  const isApiRoute = pathname.startsWith("/api");

  if (isApiRoute) {
    if (pathname.startsWith("/api/auth") || pathname.startsWith("/api/stripe/webhook") || pathname.startsWith("/api/cron") || pathname.startsWith("/api/public")) return NextResponse.next();
    if (!req.auth) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.next();
  }

  if (!req.auth && !isAuthPage && pathname !== "/" && !pathname.startsWith("/listen")) {
    return NextResponse.redirect(new URL("/login", req.url));
  }

  if (req.auth && isAuthPage) {
    return NextResponse.redirect(new URL("/discuss", req.url));
  }

  return NextResponse.next();
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|logo.png|icon.png|manifest.json|examples).*)"],
};
