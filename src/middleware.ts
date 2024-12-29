// middleware.ts
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  // Only check for auth_token cookie, not localStorage
  const token = request.cookies.get("auth_token");
  const isAuthPage = request.nextUrl.pathname === "/login" || request.nextUrl.pathname === "/register";
  
  // Protect dashboard routes
  if (request.nextUrl.pathname.startsWith("/dashboard")) {
    if (!token) {
      return NextResponse.redirect(new URL("/login", request.url));
    }
  }

  // Redirect authenticated users away from auth pages
  if (token && isAuthPage) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/login",
    "/register",
  ],
};