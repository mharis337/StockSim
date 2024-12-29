import React from "react";

export default function Login() {
  return (
    <div className="flex h-screen">
      {/* Left Side: Image */}
      <div
        className="w-1/2 bg-cover bg-center"
        style={{ backgroundImage: `url('/fin.jpg')` }}
      >
        {/* Use an appropriate path for your image */}
      </div>

      {/* Right Side: Login Form */}
      <div
        className="flex w-1/2 items-center justify-center"
        style={{
          background: "linear-gradient(to bottom, #A2D2FF, #FFC8DD)",
        }}
      >
        <div
          className="w-full max-w-md p-8 rounded-lg shadow-lg"
          style={{
            backgroundColor: "rgba(255, 255, 255, 0.5)", // White with 90% transparency
            boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)", // Soft shadow
          }}
        >
          <h2 className="text-3xl font-bold text-center text-gray-700">
            Welcome To StockSim
          </h2>
          <form className="mt-6">
            <div className="mb-4">
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-600"
              >
                Email
              </label>
              <input
                type="email"
                id="email"
                className="w-full px-4 py-2 mt-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400"
                placeholder="Enter your email"
              />
            </div>
            <div className="mb-4">
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-600"
              >
                Password
              </label>
              <input
                type="password"
                id="password"
                className="w-full px-4 py-2 mt-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-blue-400"
                placeholder="Enter your password"
              />
            </div>
            <button
              type="submit"
              className="w-full px-4 py-2 mt-4 text-white bg-blue-600 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2"
            >
              Login
            </button>
          </form>
          <p className="mt-4 text-sm text-center text-gray-600">
            Don't have an account?{" "}
            <a
              href="/signup"
              className="text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-400"
            >
              Sign up
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
