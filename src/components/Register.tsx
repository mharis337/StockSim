"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Input } from "./ui/Input";
import { Button } from "./ui/Button";

export function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Registration failed");
      }

      router.push("/login");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen">
      <div
        className="w-1/2 bg-cover bg-center"
        style={{ backgroundImage: `url('/fin.jpg')` }}
      />
      <div
        className="flex w-1/2 items-center justify-center"
        style={{
          background: "linear-gradient(to bottom, #A2D2FF, #FFC8DD)",
        }}
      >
        <div className="w-full max-w-md space-y-8 p-8 rounded-lg bg-white/50 shadow-lg">
          <div>
            <h2 className="text-3xl font-bold text-center text-gray-700">
              Create an Account
            </h2>
          </div>
          
          {error && (
            <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}

          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email
              </label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="Enter your email"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Create a password"
              />
            </div>

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full"
            >
              {isLoading ? "Creating account..." : "Sign up"}
            </Button>
          </form>

          <p className="mt-4 text-center text-sm text-gray-600">
            Already have an account?{" "}
            <Link
              href="/login"
              className="text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-400"
            >
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}