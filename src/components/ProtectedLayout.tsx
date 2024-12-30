"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function ProtectedLayout({
  children,
  currentPage = "dashboard"
}: {
  children: React.ReactNode;
  currentPage?: "dashboard" | "trade";
}) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [userEmail, setUserEmail] = useState<string>("");

  const refreshToken = async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) return false;

      const response = await fetch("http://localhost:5000/api/refresh", {
        method: "POST",
        credentials: "include",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem("token", data.access_token);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error("Token refresh failed:", error);
      return false;
    }
  };

  const verifyAuth = async () => {
    try {
      const token = localStorage.getItem("token");
      const email = localStorage.getItem("userEmail");
      
      if (!token) {
        throw new Error("No token found");
      }

      setUserEmail(email || "");

      const response = await fetch("http://localhost:5000/api/protected", {
        method: "GET",
        credentials: "include",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!response.ok) {
        // Try to refresh the token if verification fails
        const refreshSuccess = await refreshToken();
        if (!refreshSuccess) {
          throw new Error("Authentication failed");
        }
      }

      setIsLoading(false);
    } catch (error) {
      console.error("Auth verification failed:", error);
      localStorage.removeItem("token");
      localStorage.removeItem("userEmail");
      document.cookie = "auth_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
      router.push("/login");
    }
  };

  useEffect(() => {
    verifyAuth();
    
    // Set up periodic verification and refresh
    const verifyInterval = setInterval(verifyAuth, 60000); // Check every minute
    const refreshInterval = setInterval(refreshToken, 25 * 60 * 1000); // Refresh every 25 minutes

    return () => {
      clearInterval(verifyInterval);
      clearInterval(refreshInterval);
    };
  }, [router]);

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem("token");
      
      const response = await fetch("http://localhost:5000/api/logout", {
        method: "POST",
        credentials: "include",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      // Clear auth data regardless of response
      localStorage.removeItem("token");
      localStorage.removeItem("userEmail");
      document.cookie = "auth_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
      
      router.push("/login");
    } catch (err) {
      console.error("Logout error:", err);
      // Clear auth data and redirect even if logout fails
      localStorage.removeItem("token");
      localStorage.removeItem("userEmail");
      document.cookie = "auth_token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
      router.push("/login");
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link href="/dashboard" className="text-xl font-bold text-blue-600">
                  StockSim
                </Link>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                <Link
                  href="/dashboard"
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                    currentPage === "dashboard"
                      ? "border-blue-500 text-gray-900"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  Dashboard
                </Link>
                <Link
                  href="/trade"
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                    currentPage === "trade"
                      ? "border-blue-500 text-gray-900"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  Trade
                </Link>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {userEmail && (
                <span className="text-sm text-gray-600">{userEmail}</span>
              )}
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {children}
        </div>
      </main>
    </div>
  );
}