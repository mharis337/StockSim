// src/app/page.tsx
import Link from "next/link";
import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto pt-20 pb-16 px-4 sm:px-6 lg:px-8">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8">
              <div className="text-center">
                <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span className="block">Master Stock Trading</span>
                  <span className="block text-blue-600">Without the Risk</span>
                </h1>
                <p className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
                  Practice trading with real-time market data. Build your confidence
                  and strategy in a risk-free environment.
                </p>
                <div className="mt-5 max-w-md mx-auto sm:flex sm:justify-center md:mt-8">
                  <div className="rounded-md shadow">
                    <Link
                      href="/register"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 md:py-4 md:text-lg md:px-10"
                    >
                      Get Started
                    </Link>
                  </div>
                  <div className="mt-3 sm:mt-0 sm:ml-3">
                    <Link
                      href="/login"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-blue-600 bg-white hover:bg-gray-50 md:py-4 md:text-lg md:px-10"
                    >
                      Sign In
                    </Link>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900">
              Why Choose StockSim?
            </h2>
          </div>

          <div className="mt-16">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
              {/* Feature 1 */}
              <div className="pt-6">
                <div className="flow-root rounded-lg bg-gray-50 px-6 pb-8">
                  <div className="-mt-6">
                    <div>
                      <span className="inline-flex items-center justify-center rounded-md bg-blue-500 p-3 shadow-lg">
                        <Image
                          src="/globe.svg"
                          alt="Real-time data"
                          width={24}
                          height={24}
                          className="h-6 w-6 text-white"
                        />
                      </span>
                    </div>
                    <h3 className="mt-8 text-lg font-medium tracking-tight text-gray-900">
                      Real-Time Market Data
                    </h3>
                    <p className="mt-5 text-base text-gray-500">
                      Practice with live market conditions using real-time stock data
                      and market indicators.
                    </p>
                  </div>
                </div>
              </div>

              {/* Feature 2 */}
              <div className="pt-6">
                <div className="flow-root rounded-lg bg-gray-50 px-6 pb-8">
                  <div className="-mt-6">
                    <div>
                      <span className="inline-flex items-center justify-center rounded-md bg-blue-500 p-3 shadow-lg">
                        <Image
                          src="/file.svg"
                          alt="Portfolio tracking"
                          width={24}
                          height={24}
                          className="h-6 w-6 text-white"
                        />
                      </span>
                    </div>
                    <h3 className="mt-8 text-lg font-medium tracking-tight text-gray-900">
                      Portfolio Tracking
                    </h3>
                    <p className="mt-5 text-base text-gray-500">
                      Monitor your virtual portfolio performance and track your
                      trading history.
                    </p>
                  </div>
                </div>
              </div>

              {/* Feature 3 */}
              <div className="pt-6">
                <div className="flow-root rounded-lg bg-gray-50 px-6 pb-8">
                  <div className="-mt-6">
                    <div>
                      <span className="inline-flex items-center justify-center rounded-md bg-blue-500 p-3 shadow-lg">
                        <Image
                          src="/window.svg"
                          alt="Advanced analysis"
                          width={24}
                          height={24}
                          className="h-6 w-6 text-white"
                        />
                      </span>
                    </div>
                    <h3 className="mt-8 text-lg font-medium tracking-tight text-gray-900">
                      Advanced Analysis
                    </h3>
                    <p className="mt-5 text-base text-gray-500">
                      Access advanced charting tools and technical indicators for
                      better decision making.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-blue-600">
        <div className="max-w-2xl mx-auto text-center py-16 px-4 sm:py-20 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
            <span className="block">Ready to start trading?</span>
            <span className="block">Create your account today.</span>
          </h2>
          <p className="mt-4 text-lg leading-6 text-blue-100">
            Join thousands of traders learning the market with StockSim.
          </p>
          <Link
            href="/register"
            className="mt-8 w-full inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-blue-600 bg-white hover:bg-blue-50 sm:w-auto"
          >
            Get Started for Free
          </Link>
        </div>
      </div>
    </div>
  );
}