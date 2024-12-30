// app/page.tsx
'use client';

export default function Page() {
  return (
    <div style={{ minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ backgroundColor: '#CDB4DB' }}>
        <div
          style={{
            maxWidth: '1200px',
            margin: '0 auto',
            padding: '8px 16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            color: 'white',
          }}
        >
          <div
            style={{
              height: '50px',
              width: '180px',
              position: 'relative',
            }}
          >
            <img
              src="/logo.png"
              alt="StockSim Logo"
              style={{
                height: '100%',
                width: '100%',
                objectFit: 'contain',
              }}
            />
          </div>
          {/* Removed the buttons from the header */}
          {/* <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
            <a href="/register" style={{ 
              padding: '6px 16px', 
              backgroundColor: 'white', 
              color: '#CDB4DB', 
              borderRadius: '6px', 
              fontWeight: '600'
            }}>
              Get Started
            </a>
            <a href="/login" style={{ 
              color: 'white',
              textShadow: '1px 1px 2px rgba(0,0,0,0.1)',
              fontWeight: '500'
            }}>
              Sign In
            </a>
          </div> */}
        </div>
      </div>

      {/* Hero Section */}
      <div
        style={{
          backgroundColor: '#CDB4DB',
          position: 'relative',
          overflow: 'hidden',
          paddingBottom: '100px',
        }}
      >
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 16px' }}>
          <div
            style={{
              textAlign: 'center',
              padding: '80px 0 60px',
              maxWidth: '800px',
              margin: '0 auto',
            }}
          >
            <h1
              style={{
                fontSize: '3.5rem',
                fontWeight: 'bold',
                color: 'white',
                marginBottom: '24px',
                lineHeight: '1.1',
                textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
              }}
            >
              <span style={{ display: 'block' }}>Master Stock Trading</span>
              <span
                style={{
                  display: 'block',
                  color: '#FFAFCC',
                  textShadow: '2px 2px 4px rgba(0,0,0,0.15)',
                }}
              >
                Without the Risk
              </span>
            </h1>
            <p
              style={{
                fontSize: '1.25rem',
                color: 'white',
                marginBottom: '32px',
                textShadow: '1px 1px 2px rgba(0,0,0,0.1)',
                fontWeight: '500',
              }}
            >
              Practice trading with real-time market data. Build your confidence
              and strategy in a risk-free environment.
            </p>
            <div
              style={{
                display: 'flex',
                gap: '16px',
                justifyContent: 'center',
                flexWrap: 'wrap', // Ensures responsiveness on smaller screens
              }}
            >
              <a
                href="/register"
                style={{
                  display: 'inline-block', // Changed from default to inline-block
                  padding: '12px 32px',
                  backgroundColor: '#A2D2FF',
                  color: 'white',
                  borderRadius: '8px',
                  fontWeight: '600',
                  textDecoration: 'none',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  transition: 'background-color 0.3s ease',
                  cursor: 'pointer', // Added cursor pointer
                  textAlign: 'center', // Ensures text is centered
                }}
              >
                Get Started
              </a>
              <a
                href="/login"
                style={{
                  display: 'inline-block', // Changed from default to inline-block
                  padding: '12px 32px',
                  backgroundColor: 'white', // Changed background to white for consistency
                  color: '#A2D2FF',
                  borderRadius: '8px',
                  fontWeight: '600',
                  textDecoration: 'none',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  transition: 'background-color 0.3s ease',
                  cursor: 'pointer', // Added cursor pointer
                  textAlign: 'center', // Ensures text is centered
                }}
              >
                Sign In
              </a>
            </div>
          </div>
        </div>

        {/* Wave SVG */}
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            width: '100%',
            height: '200px',
            overflow: 'hidden',
            pointerEvents: 'none', // Prevents SVG from capturing mouse events
          }}
        >
          <svg
            viewBox="0 0 1440 390"
            preserveAspectRatio="none"
            style={{
              position: 'absolute',
              bottom: 0,
              width: '100%',
              height: '100%',
              transform: 'scale(1.2)',
            }}
          >
            <path
              fill="white"
              d="M0,192L40,181.3C80,171,160,149,240,154.7C320,160,400,192,480,186.7C560,181,640,139,720,144C800,149,880,203,960,208C1040,213,1120,171,1200,160C1280,149,1360,171,1400,181.3L1440,192L1440,390L1400,390C1360,390,1280,390,1200,390C1120,390,1040,390,960,390C880,390,800,390,720,390C640,390,560,390,480,390C400,390,320,390,240,390C160,390,80,390,40,390L0,390Z"
            />
            <path
              fill="white"
              fillOpacity="0.8"
              d="M0,224L40,229.3C80,235,160,245,240,240C320,235,400,213,480,202.7C560,192,640,192,720,197.3C800,203,880,213,960,229.3C1040,245,1120,267,1200,261.3C1280,256,1360,224,1400,208L1440,192L1440,390L1400,390C1360,390,1280,390,1200,390C1120,390,1040,390,960,390C880,390,800,390,720,390C640,390,560,390,480,390C400,390,320,390,240,390C160,390,80,390,40,390L0,390Z"
            />
          </svg>
        </div>
      </div>

      {/* Features Section */}
      <div style={{ padding: '96px 0', backgroundColor: 'white' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 16px' }}>
          <h2
            style={{
              fontSize: '2rem',
              fontWeight: 'bold',
              textAlign: 'center',
              marginBottom: '64px',
              color: '#1F2937',
            }}
          >
            Why Choose StockSim?
          </h2>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', // Improved responsiveness
              gap: '32px',
            }}
          >
            {/* Feature Cards */}
            {[
              {
                icon: '📊',
                title: 'Real-Time Market Data',
                description:
                  'Practice with live market conditions using real-time stock data and market indicators.',
                bgColor: '#BDE0FE',
              },
              {
                icon: '📈',
                title: 'Portfolio Tracking',
                description:
                  'Monitor your virtual portfolio performance and track your trading history.',
                bgColor: '#FFAFCC',
              },
              {
                icon: '🤖',
                title: 'AI Trading Models',
                description:
                  'Create and backtest ML trading models. Compare performance across different strategies and market conditions.',
                bgColor: '#A2D2FF',
              },
            ].map((feature, index) => (
              <div
                key={index}
                style={{
                  backgroundColor: '#F9FAFB',
                  borderRadius: '8px',
                  padding: '32px',
                  position: 'relative',
                  paddingTop: '48px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    top: '-24px',
                    left: '32px',
                    backgroundColor: feature.bgColor,
                    padding: '12px',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '48px',
                    height: '48px',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  }}
                >
                  {feature.icon}
                </div>
                <h3
                  style={{
                    fontSize: '1.25rem',
                    fontWeight: 'bold',
                    marginBottom: '16px',
                    color: '#1F2937',
                  }}
                >
                  {feature.title}
                </h3>
                <p
                  style={{
                    color: '#4B5563',
                    lineHeight: '1.5',
                  }}
                >
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div
        style={{
          backgroundColor: '#CDB4DB',
          padding: '64px 0',
          textAlign: 'center',
        }}
      >
        <div style={{ maxWidth: '800px', margin: '0 auto', padding: '0 16px' }}>
          <h2
            style={{
              fontSize: '2.5rem',
              fontWeight: 'bold',
              marginBottom: '16px',
              color: 'white',
              textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
            }}
          >
            Ready to start trading?
            <br />
            Create your account today.
          </h2>
          <p
            style={{
              fontSize: '1.125rem',
              marginBottom: '32px',
              color: 'white',
              textShadow: '1px 1px 2px rgba(0,0,0,0.1)',
              fontWeight: '500',
            }}
          >
            Join thousands of traders learning the market with StockSim.
          </p>
          <a
            href="/register"
            style={{
              display: 'inline-block',
              padding: '12px 32px',
              backgroundColor: 'white',
              color: '#CDB4DB',
              borderRadius: '8px',
              fontWeight: '600',
              textDecoration: 'none',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              transition: 'background-color 0.3s ease',
              cursor: 'pointer', // Added cursor pointer
              textAlign: 'center', // Ensures text is centered
            }}
          >
            Get Started for Free
          </a>
        </div>
      </div>
    </div>
  );
}
