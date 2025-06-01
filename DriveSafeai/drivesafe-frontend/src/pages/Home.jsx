import React, { useState, useEffect } from 'react';

const DriveSafeHomepage = () => {
  const [particles, setParticles] = useState([]);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Generate particles
    const newParticles = [];
    for (let i = 0; i < 30; i++) {
      newParticles.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        delay: Math.random() * 3,
        duration: 3 + Math.random() * 2
      });
    }
    setParticles(newParticles);

    // Mouse tracking
    const handleMouseMove = (e) => {
      setMousePos({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleButtonClick = (path) => {
    console.log(`Navigate to: ${path}`);
    // Add your navigation logic here
  };

  const containerStyle = {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    position: 'relative',
    overflow: 'hidden'
  };

  const particleStyle = (particle) => ({
    position: 'absolute',
    left: `${particle.x}%`,
    top: `${particle.y}%`,
    width: '3px',
    height: '3px',
    backgroundColor: 'rgba(255, 255, 255, 0.4)',
    borderRadius: '50%',
    animation: `float ${particle.duration}s ease-in-out infinite`,
    animationDelay: `${particle.delay}s`
  });

  const heroSectionStyle = {
    display: 'flex',
    alignItems: 'center',
    minHeight: '100vh',
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem',
    gap: '3rem',
    position: 'relative',
    zIndex: 2
  };

  const heroContentStyle = {
    flex: 1,
    color: 'white',
    animation: 'slideInLeft 1s ease-out'
  };

  const heroVisualStyle = {
    flex: 1,
    animation: 'slideInRight 1s ease-out'
  };

  const titleStyle = {
    fontSize: 'clamp(2.5rem, 5vw, 4rem)',
    fontWeight: '700',
    marginBottom: '1.5rem',
    textShadow: '0 4px 20px rgba(0,0,0,0.3)',
    lineHeight: '1.1'
  };

  const subtitleStyle = {
    fontSize: '1.25rem',
    marginBottom: '2rem',
    opacity: 0.9,
    lineHeight: '1.6'
  };

  const buttonContainerStyle = {
    display: 'flex',
    gap: '1rem',
    marginBottom: '2rem',
    flexWrap: 'wrap'
  };

  const primaryButtonStyle = {
    padding: '1rem 2rem',
    backgroundColor: '#ff6b6b',
    color: 'white',
    border: 'none',
    borderRadius: '50px',
    fontSize: '1.1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: '0 8px 25px rgba(255, 107, 107, 0.3)',
    transform: 'translateY(0)',
    outline: 'none'
  };

  const secondaryButtonStyle = {
    padding: '1rem 2rem',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    color: 'white',
    border: '2px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '50px',
    fontSize: '1.1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    backdropFilter: 'blur(10px)',
    transform: 'translateY(0)',
    outline: 'none'
  };

  const featuresStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '1.5rem',
    marginTop: '2rem'
  };

  const featureCardStyle = {
    padding: '1.5rem',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '20px',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    textAlign: 'center',
    transition: 'transform 0.3s ease',
    cursor: 'pointer'
  };

  const featureIconStyle = {
    width: '60px',
    height: '60px',
    backgroundColor: '#4facfe',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    margin: '0 auto 1rem',
    fontSize: '1.5rem',
    animation: 'pulse 2s infinite'
  };

  const heroImageStyle = {
    width: '100%',
    height: '400px',
    background: 'linear-gradient(45deg, #667eea, #764ba2)',
    borderRadius: '20px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '1.2rem',
    textAlign: 'center',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    transition: 'transform 0.3s ease',
    position: 'relative',
    overflow: 'hidden'
  };

  const floatingElementStyle = (index) => ({
    position: 'absolute',
    width: '30px',
    height: '30px',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '50%',
    backdropFilter: 'blur(5px)',
    transform: `translate(${mousePos.x * (index + 1) * 0.1}px, ${mousePos.y * (index + 1) * 0.1}px)`,
    transition: 'transform 0.3s ease',
    animation: `floatAround ${8 + index}s ease-in-out infinite`,
    animationDelay: `${index * 2}s`
  });

  return (
      <div style={containerStyle}>
        <style>
          {`
          @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.4; }
            50% { transform: translateY(-15px) rotate(180deg); opacity: 0.8; }
          }
          
          @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
          }
          
          @keyframes slideInRight {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
          }
          
          @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
          }
          
          @keyframes floatAround {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(10px, -10px) rotate(90deg); }
            50% { transform: translate(-5px, -15px) rotate(180deg); }
            75% { transform: translate(-8px, 8px) rotate(270deg); }
          }
          
          @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-8px); }
            60% { transform: translateY(-4px); }
          }
          
          .btn-hover:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 35px rgba(255, 107, 107, 0.4) !important;
          }
          
          .btn-secondary-hover:hover {
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.5) !important;
            transform: translateY(-3px) !important;
          }
          
          .feature-hover:hover {
            transform: translateY(-5px) !important;
          }
          
          .hero-image-hover:hover {
            transform: scale(1.02) !important;
          }
          
          .car-bounce {
            animation: bounce 2s infinite;
          }
          
          @media (max-width: 768px) {
            .hero-responsive {
              flex-direction: column;
              text-align: center;
              gap: 2rem;
            }
            .button-responsive {
              flex-direction: column;
              align-items: center;
            }
            .button-responsive button {
              width: 200px;
            }
          }
        `}
        </style>

        {/* Background Particles */}
        {particles.map((particle) => (
            <div
                key={particle.id}
                style={particleStyle(particle)}
            />
        ))}

        {/* Floating Elements */}
        <div style={{...floatingElementStyle(0), top: '20%', left: '10%'}} />
        <div style={{...floatingElementStyle(1), top: '60%', right: '15%'}} />
        <div style={{...floatingElementStyle(2), bottom: '30%', left: '5%'}} />

        {/* Main Content */}
        <section style={heroSectionStyle} className="hero-responsive">
          {/* Left Content */}
          <div style={heroContentStyle}>
            <h1 style={titleStyle}>
              Welcome to DriveSafeAI
            </h1>

            <p style={subtitleStyle}>
              Revolutionizing road safety with cutting-edge AI technology. Our platform provides intelligent risk assessment, real-time safety monitoring, and smart insurance management to keep you protected on every journey.
            </p>

            <div style={buttonContainerStyle} className="button-responsive">
              <button
                  style={primaryButtonStyle}
                  className="btn-hover"
                  onClick={() => handleButtonClick('/login')}
                  onMouseEnter={(e) => {
                    e.target.style.transform = 'translateY(-3px)';
                    e.target.style.boxShadow = '0 12px 35px rgba(255, 107, 107, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = '0 8px 25px rgba(255, 107, 107, 0.3)';
                  }}
              >
                Get Started
              </button>

              <button
                  style={secondaryButtonStyle}
                  className="btn-secondary-hover"
                  onClick={() => handleButtonClick('/register')}
                  onMouseEnter={(e) => {
                    e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                    e.target.style.borderColor = 'rgba(255, 255, 255, 0.5)';
                    e.target.style.transform = 'translateY(-3px)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                    e.target.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                    e.target.style.transform = 'translateY(0)';
                  }}
              >
                Learn More
              </button>
            </div>

            {/* Features */}
            <div style={featuresStyle}>
              <div
                  style={featureCardStyle}
                  className="feature-hover"
                  onMouseEnter={(e) => e.target.style.transform = 'translateY(-5px)'}
                  onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
              >
                <div style={featureIconStyle}>🚗</div>
                <h3 style={{margin: '0 0 0.5rem 0', fontSize: '1.1rem'}}>Smart Monitoring</h3>
                <p style={{margin: 0, opacity: 0.8, fontSize: '0.9rem'}}>Real-time driving analysis and safety alerts</p>
              </div>

              <div
                  style={featureCardStyle}
                  className="feature-hover"
                  onMouseEnter={(e) => e.target.style.transform = 'translateY(-5px)'}
                  onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
              >
                <div style={featureIconStyle}>🧠</div>
                <h3 style={{margin: '0 0 0.5rem 0', fontSize: '1.1rem'}}>AI Risk Assessment</h3>
                <p style={{margin: 0, opacity: 0.8, fontSize: '0.9rem'}}>Predictive safety scoring and recommendations</p>
              </div>

              <div
                  style={featureCardStyle}
                  className="feature-hover"
                  onMouseEnter={(e) => e.target.style.transform = 'translateY(-5px)'}
                  onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
              >
                <div style={featureIconStyle}>🛡️</div>
                <h3 style={{margin: '0 0 0.5rem 0', fontSize: '1.1rem'}}>Insurance Integration</h3>
                <p style={{margin: 0, opacity: 0.8, fontSize: '0.9rem'}}>Seamless policy management and claims</p>
              </div>
            </div>
          </div>

          {/* Right Visual */}
          <div style={heroVisualStyle}>
            <div
                style={heroImageStyle}
                className="hero-image-hover"
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.02)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
            >
              <div>
                <div style={{fontSize: '4rem', marginBottom: '1rem'}} className="car-bounce">
                  🤖
                </div>
                <div style={{lineHeight: '1.4'}}>
                  AI-Powered<br />
                  Driving Safety<br />
                  Technology
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
  );
};

export default DriveSafeHomepage;