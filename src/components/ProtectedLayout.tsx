"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function ProtectedLayout({
  children,
  currentPage = "dashboard"
}: {
  children: React.ReactNode;
  currentPage?: "dashboard" | "trade" | "model-training";
}) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [userEmail, setUserEmail] = useState<string>("");

  useEffect(() => {
    const email = localStorage.getItem("userEmail");
    if (email) setUserEmail(email);
  }, []);

  const verifyAuth = async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("No token found");
      }

      const response = await fetch("http://localhost:5000/api/protected", {
        method: "GET",
        credentials: "include",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!response.ok) {
        throw new Error("Authentication failed");
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
  }, []);

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:5000/api/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch (err) {
      console.error("Logout error:", err);
    } finally {
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
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#CDB4DB] mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <nav style={{ backgroundColor: '#CDB4DB' }} className="shadow-md">
        <div className="px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center">
              <Link href="/dashboard" className="flex items-center gap-2">
                <img src="/logo.png" alt="StockSim Logo" className="h-8 w-auto" />
                <span className="text-xl font-bold text-white">StockSim</span>
              </Link>
            </div>

            {/* Navigation and User Info */}
            <div className="flex items-center gap-6">
              {/* Navigation Links */}
              <div className="flex items-center gap-4">
                <Link 
                  href="/dashboard"
                  style={{ 
                    backgroundColor: currentPage === "dashboard" ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
                    color: 'white'
                  }}
                  className="px-3 py-2 rounded-md text-sm font-medium hover:bg-white/10 transition-colors"
                >
                  Dashboard
                </Link>
                <Link 
                  href="/trade"
                  style={{ 
                    backgroundColor: currentPage === "trade" ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
                    color: 'white'
                  }}
                  className="px-3 py-2 rounded-md text-sm font-medium hover:bg-white/10 transition-colors"
                >
                  Trade
                </Link>
                <Link 
                  href="/model-training"
                  style={{ 
                    backgroundColor: currentPage === "model-training" ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
                    color: 'white'
                  }}
                  className="px-3 py-2 rounded-md text-sm font-medium hover:bg-white/10 transition-colors"
                >
                  AI Models
                </Link>
              </div>
              
              <div className="flex items-center gap-4">
                {userEmail && (
                  <span className="text-white text-sm hidden sm:block">
                    {userEmail}
                  </span>
                )}
                
                <button
                  onClick={handleLogout}
                  style={{ backgroundColor: '#FFAFCC' }}
                  className="text-white hover:bg-opacity-90 px-4 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <div className="flex-1">
        {children}
      </div>
    </div>
  );
}