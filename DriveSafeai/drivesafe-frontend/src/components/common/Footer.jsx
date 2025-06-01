import React, { useState } from 'react';

const Footer = () => {
    const year = new Date().getFullYear();
    const [hoveredLink, setHoveredLink] = useState(null);

    const footerStyle = {
        background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        color: '#e2e8f0',
        marginTop: 'auto',
        position: 'relative',
        overflow: 'hidden',
        animation: 'slideUp 0.8s ease-out'
    };

    const containerStyle = {
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '2rem 1.5rem',
        position: 'relative',
        zIndex: 10
    };

    const brandStyle = {
        fontSize: '1.5rem',
        fontWeight: 'bold',
        marginBottom: '0.5rem',
        transition: 'transform 0.3s ease',
        cursor: 'pointer'
    };

    const brandHoverStyle = {
        transform: 'scale(1.05)'
    };

    const aiTextStyle = {
        background: 'linear-gradient(45deg, #3b82f6, #06b6d4)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        backgroundClip: 'text'
    };

    const linkStyle = {
        color: '#94a3b8',
        textDecoration: 'none',
        padding: '0.5rem 1rem',
        position: 'relative',
        transition: 'all 0.3s ease',
        borderRadius: '4px'
    };

    const linkHoverStyle = {
        color: '#ffffff',
        background: 'rgba(59, 130, 246, 0.1)',
        transform: 'translateY(-2px)'
    };

    const backgroundDotStyle = {
        position: 'absolute',
        width: '4px',
        height: '4px',
        background: 'rgba(59, 130, 246, 0.3)',
        borderRadius: '50%',
        animation: 'float 3s ease-in-out infinite'
    };

    const pulseStyle = {
        animation: 'pulse 2s ease-in-out infinite'
    };

    return (
        <>
            <style>{`
                @keyframes slideUp {
                    from {
                        transform: translateY(20px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                
                @keyframes float {
                    0%, 100% { transform: translateY(0px); opacity: 0.3; }
                    50% { transform: translateY(-10px); opacity: 0.8; }
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 0.4; }
                    50% { opacity: 1; }
                }
                
                @keyframes shimmer {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }
            `}</style>

            <footer style={footerStyle}>
                {/* Floating background dots */}
                {[...Array(8)].map((_, i) => (
                    <div
                        key={i}
                        style={{
                            ...backgroundDotStyle,
                            top: `${20 + (i * 10)}%`,
                            left: `${10 + (i * 11)}%`,
                            animationDelay: `${i * 0.5}s`
                        }}
                    />
                ))}

                <div style={containerStyle}>
                    {/* Main content row */}
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '2rem'
                    }}>

                        {/* Brand section */}
                        <div
                            style={{
                                textAlign: window.innerWidth < 768 ? 'center' : 'left',
                                flex: '1'
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.transform = 'scale(1.05)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.transform = 'scale(1)';
                            }}
                        >
                            <div style={{
                                ...brandStyle,
                                transition: 'transform 0.3s ease'
                            }}>
                                <span>DriveSafe</span>
                                <span style={aiTextStyle}>AI</span>
                            </div>
                            <p style={{
                                color: '#64748b',
                                fontSize: '0.875rem',
                                margin: 0,
                                ...pulseStyle
                            }}>
                                © {year} All rights reserved
                            </p>
                        </div>

                        {/* Navigation links */}
                        <div style={{
                            display: 'flex',
                            gap: '0.5rem',
                            flexWrap: 'wrap',
                            justifyContent: 'center'
                        }}>
                            {[
                                { name: 'Privacy Policy', href: '#privacy' },
                                { name: 'Terms of Service', href: '#terms' },
                                { name: 'Contact Us', href: '#contact' }
                            ].map((link, index) => (
                                <a
                                    key={link.name}
                                    href={link.href}
                                    style={{
                                        ...linkStyle,
                                        ...(hoveredLink === index ? linkHoverStyle : {})
                                    }}
                                    onMouseEnter={() => setHoveredLink(index)}
                                    onMouseLeave={() => setHoveredLink(null)}
                                >
                                    {link.name}
                                    {hoveredLink === index && (
                                        <div style={{
                                            position: 'absolute',
                                            bottom: '0',
                                            left: '50%',
                                            transform: 'translateX(-50%)',
                                            width: '80%',
                                            height: '2px',
                                            background: 'linear-gradient(45deg, #3b82f6, #06b6d4)',
                                            borderRadius: '1px',
                                            animation: 'shimmer 0.6s ease-out'
                                        }} />
                                    )}
                                </a>
                            ))}
                        </div>
                    </div>

                    {/* Bottom section */}
                    <div style={{
                        marginTop: '2rem',
                        paddingTop: '1.5rem',
                        borderTop: '1px solid #334155',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '1rem'
                    }}>
                        <p style={{
                            color: '#64748b',
                            fontSize: '0.75rem',
                            margin: 0,
                            textTransform: 'uppercase',
                            letterSpacing: '0.1em'
                        }}>
                            Powered by Artificial Intelligence
                        </p>

                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            {[...Array(4)].map((_, i) => (
                                <div
                                    key={i}
                                    style={{
                                        width: '8px',
                                        height: '8px',
                                        background: '#64748b',
                                        borderRadius: '50%',
                                        transition: 'all 0.3s ease',
                                        cursor: 'pointer',
                                        animation: `pulse 2s ease-in-out infinite`,
                                        animationDelay: `${i * 0.3}s`
                                    }}
                                    onMouseEnter={(e) => {
                                        e.target.style.background = '#3b82f6';
                                        e.target.style.transform = 'scale(1.5)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.target.style.background = '#64748b';
                                        e.target.style.transform = 'scale(1)';
                                    }}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </footer>
        </>
    );
};

export default Footer;