import React, { useState, useEffect } from 'react';
import { Shield, Calendar, DollarSign, FileText, TrendingUp, Award } from 'lucide-react';
import UserService from "../../services/UserService.js";

const InsurancePage = () => {
    const [policyData, setPolicyData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [animateIn, setAnimateIn] = useState(false);

    useEffect(() => {
        // Simulate API call - replace with actual UserService.getInsurance()
        const fetchInsuranceData = async () => {
            try {
                const userId = localStorage.getItem('userId');
                const data = await UserService.getInsurance(userId);
                setPolicyData(data);
                setTimeout(() => {
                    setPolicyData(data);
                    setLoading(false);
                    setAnimateIn(true);
                }, 1000);
            } catch (error) {
                console.error('Error fetching insurance data:', error);
                setLoading(false);
            }
        };

        fetchInsuranceData();
    }, []);

    const formatCurrency = (amount) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 2
        }).format(amount);
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    };

    const calculateDaysRemaining = (endDate) => {
        const today = new Date();
        const end = new Date(endDate);
        const diffTime = end - today;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        return diffDays;
    };

    if (loading) {
        return (
            <div style={{
                minHeight: '100vh',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            }}>
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '20px'
                }}>
                    <div style={{
                        width: '60px',
                        height: '60px',
                        border: '4px solid rgba(255,255,255,0.3)',
                        borderTop: '4px solid white',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite'
                    }}></div>
                    <p style={{ color: 'white', fontSize: '18px', margin: 0 }}>Loading your policy details...</p>
                </div>
                <style>
                    {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
                </style>
            </div>
        );
    }

    if (!policyData) {
        return (
            <div style={{
                minHeight: '100vh',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            }}>
                <div style={{
                    background: 'rgba(255,255,255,0.1)',
                    backdropFilter: 'blur(10px)',
                    padding: '40px',
                    borderRadius: '20px',
                    textAlign: 'center',
                    color: 'white'
                }}>
                    <h2 style={{ margin: '0 0 10px 0' }}>Unable to load policy data</h2>
                    <p style={{ margin: 0, opacity: 0.8 }}>Please try again later</p>
                </div>
            </div>
        );
    }

    const daysRemaining = calculateDaysRemaining(policyData.policyEndDate);

    return (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            padding: '20px'
        }}>
            <div style={{
                maxWidth: '1200px',
                margin: '0 auto',
                transform: animateIn ? 'translateY(0)' : 'translateY(30px)',
                opacity: animateIn ? 1 : 0,
                transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
            }}>
                {/* Header */}
                <div style={{
                    textAlign: 'center',
                    marginBottom: '40px',
                    color: 'white'
                }}>
                    <div style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '15px',
                        marginBottom: '20px'
                    }}>
                        <Shield size={40} style={{ color: '#ffd700' }} />
                        <h1 style={{
                            margin: 0,
                            fontSize: '2.5rem',
                            fontWeight: '700',
                            background: 'linear-gradient(45deg, #ffd700, #ffed4e)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            backgroundClip: 'text'
                        }}>
                            Insurance Dashboard
                        </h1>
                    </div>
                    <p style={{
                        margin: 0,
                        fontSize: '1.1rem',
                        opacity: 0.9,
                        fontWeight: '300'
                    }}>
                        Your comprehensive policy overview
                    </p>
                </div>

                {/* Status Card */}
                <div style={{
                    background: daysRemaining > 30 ? 'linear-gradient(135deg, #4ade80, #22c55e)' :
                        daysRemaining > 7 ? 'linear-gradient(135deg, #fbbf24, #f59e0b)' :
                            'linear-gradient(135deg, #ef4444, #dc2626)',
                    borderRadius: '20px',
                    padding: '25px',
                    marginBottom: '30px',
                    color: 'white',
                    textAlign: 'center',
                    boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
                    transform: 'translateY(0)',
                    transition: 'transform 0.3s ease, box-shadow 0.3s ease'
                }}
                     onMouseEnter={(e) => {
                         e.currentTarget.style.transform = 'translateY(-5px)';
                         e.currentTarget.style.boxShadow = '0 25px 50px rgba(0,0,0,0.15)';
                     }}
                     onMouseLeave={(e) => {
                         e.currentTarget.style.transform = 'translateY(0)';
                         e.currentTarget.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
                     }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', marginBottom: '10px' }}>
                        <Calendar size={24} />
                        <h3 style={{ margin: 0, fontSize: '1.3rem' }}>Policy Status</h3>
                    </div>
                    <p style={{ margin: 0, fontSize: '1.1rem', opacity: 0.9 }}>
                        {daysRemaining > 0 ? `${daysRemaining} days remaining` : 'Policy Expired'}
                    </p>
                </div>

                {/* Main Grid */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
                    gap: '25px',
                    marginBottom: '30px'
                }}>
                    {/* Policy Information Card */}
                    <div style={{
                        background: 'rgba(255, 255, 255, 0.95)',
                        backdropFilter: 'blur(20px)',
                        borderRadius: '20px',
                        padding: '30px',
                        boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        transform: 'translateY(0)',
                        transition: 'transform 0.3s ease, box-shadow 0.3s ease'
                    }}
                         onMouseEnter={(e) => {
                             e.currentTarget.style.transform = 'translateY(-5px)';
                             e.currentTarget.style.boxShadow = '0 25px 50px rgba(0,0,0,0.15)';
                         }}
                         onMouseLeave={(e) => {
                             e.currentTarget.style.transform = 'translateY(0)';
                             e.currentTarget.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
                         }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
                            <div style={{
                                background: 'linear-gradient(135deg, #667eea, #764ba2)',
                                borderRadius: '12px',
                                padding: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                <FileText size={24} style={{ color: 'white' }} />
                            </div>
                            <h2 style={{ margin: 0, color: '#1f2937', fontSize: '1.4rem', fontWeight: '600' }}>
                                Policy Information
                            </h2>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Policy Number
                                </label>
                                <p style={{ margin: 0, fontSize: '1.1rem', color: '#1f2937', fontWeight: '600' }}>
                                    {policyData.policyNumber}
                                </p>
                            </div>

                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Policy ID
                                </label>
                                <p style={{ margin: 0, fontSize: '1.1rem', color: '#1f2937', fontWeight: '600' }}>
                                    #{policyData.policyId}
                                </p>
                            </div>

                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Coverage Type
                                </label>
                                <p style={{ margin: 0, fontSize: '1.1rem', color: '#1f2937', fontWeight: '600' }}>
                                    {policyData.coverageType}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Financial Details Card */}
                    <div style={{
                        background: 'rgba(255, 255, 255, 0.95)',
                        backdropFilter: 'blur(20px)',
                        borderRadius: '20px',
                        padding: '30px',
                        boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        transform: 'translateY(0)',
                        transition: 'transform 0.3s ease, box-shadow 0.3s ease'
                    }}
                         onMouseEnter={(e) => {
                             e.currentTarget.style.transform = 'translateY(-5px)';
                             e.currentTarget.style.boxShadow = '0 25px 50px rgba(0,0,0,0.15)';
                         }}
                         onMouseLeave={(e) => {
                             e.currentTarget.style.transform = 'translateY(0)';
                             e.currentTarget.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
                         }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
                            <div style={{
                                background: 'linear-gradient(135deg, #10b981, #059669)',
                                borderRadius: '12px',
                                padding: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                <DollarSign size={24} style={{ color: 'white' }} />
                            </div>
                            <h2 style={{ margin: 0, color: '#1f2937', fontSize: '1.4rem', fontWeight: '600' }}>
                                Financial Details
                            </h2>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Coverage Amount
                                </label>
                                <p style={{ margin: 0, fontSize: '1.4rem', color: '#059669', fontWeight: '700' }}>
                                    {formatCurrency(policyData.covarageAmount)}
                                </p>
                            </div>

                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Base Premium
                                </label>
                                <p style={{ margin: 0, fontSize: '1.1rem', color: '#1f2937', fontWeight: '600' }}>
                                    {formatCurrency(policyData.basePremium)}
                                </p>
                            </div>

                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Final Premium
                                </label>
                                <p style={{ margin: 0, fontSize: '1.2rem', color: '#dc2626', fontWeight: '700' }}>
                                    {formatCurrency(policyData.finalPremium)}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Bottom Row */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
                    gap: '25px'
                }}>
                    {/* Policy Dates Card */}
                    <div style={{
                        background: 'rgba(255, 255, 255, 0.95)',
                        backdropFilter: 'blur(20px)',
                        borderRadius: '20px',
                        padding: '30px',
                        boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        transform: 'translateY(0)',
                        transition: 'transform 0.3s ease, box-shadow 0.3s ease'
                    }}
                         onMouseEnter={(e) => {
                             e.currentTarget.style.transform = 'translateY(-5px)';
                             e.currentTarget.style.boxShadow = '0 25px 50px rgba(0,0,0,0.15)';
                         }}
                         onMouseLeave={(e) => {
                             e.currentTarget.style.transform = 'translateY(0)';
                             e.currentTarget.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
                         }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
                            <div style={{
                                background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                                borderRadius: '12px',
                                padding: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                <Calendar size={24} style={{ color: 'white' }} />
                            </div>
                            <h2 style={{ margin: 0, color: '#1f2937', fontSize: '1.4rem', fontWeight: '600' }}>
                                Policy Period
                            </h2>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    Start Date
                                </label>
                                <p style={{ margin: 0, fontSize: '1.1rem', color: '#1f2937', fontWeight: '600' }}>
                                    {formatDate(policyData.policyStartDate)}
                                </p>
                            </div>

                            <div>
                                <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                    End Date
                                </label>
                                <p style={{ margin: 0, fontSize: '1.1rem', color: '#1f2937', fontWeight: '600' }}>
                                    {formatDate(policyData.policyEndDate)}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Risk Score Card */}
                    <div style={{
                        background: 'rgba(255, 255, 255, 0.95)',
                        backdropFilter: 'blur(20px)',
                        borderRadius: '20px',
                        padding: '30px',
                        boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        transform: 'translateY(0)',
                        transition: 'transform 0.3s ease, box-shadow 0.3s ease'
                    }}
                         onMouseEnter={(e) => {
                             e.currentTarget.style.transform = 'translateY(-5px)';
                             e.currentTarget.style.boxShadow = '0 25px 50px rgba(0,0,0,0.15)';
                         }}
                         onMouseLeave={(e) => {
                             e.currentTarget.style.transform = 'translateY(0)';
                             e.currentTarget.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
                         }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
                            <div style={{
                                background: policyData.driscScore >= 8 ? 'linear-gradient(135deg, #10b981, #059669)' :
                                    policyData.driscScore >= 6 ? 'linear-gradient(135deg, #f59e0b, #d97706)' :
                                        'linear-gradient(135deg, #ef4444, #dc2626)',
                                borderRadius: '12px',
                                padding: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                <TrendingUp size={24} style={{ color: 'white' }} />
                            </div>
                            <h2 style={{ margin: 0, color: '#1f2937', fontSize: '1.4rem', fontWeight: '600' }}>
                                Risk Assessment
                            </h2>
                        </div>

                        <div style={{ textAlign: 'center' }}>
                            <div style={{
                                width: '120px',
                                height: '120px',
                                borderRadius: '50%',
                                background: `conic-gradient(${
                                    policyData.driscScore >= 8 ? '#10b981' :
                                        policyData.driscScore >= 6 ? '#f59e0b' :
                                            '#ef4444'
                                } ${policyData.driscScore * 36}deg, #e5e7eb 0deg)`,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                margin: '0 auto 15px auto',
                                position: 'relative'
                            }}>
                                <div style={{
                                    width: '90px',
                                    height: '90px',
                                    borderRadius: '50%',
                                    background: 'white',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    flexDirection: 'column'
                                }}>
                  <span style={{ fontSize: '1.8rem', fontWeight: '700', color: '#1f2937' }}>
                    {policyData.driscScore}
                  </span>
                                    <span style={{ fontSize: '0.8rem', color: '#6b7280' }}>/ 100</span>
                                </div>
                            </div>

                            <label style={{ display: 'block', fontSize: '0.9rem', color: '#6b7280', marginBottom: '5px', fontWeight: '500' }}>
                                DRISC Score
                            </label>
                            <p style={{
                                margin: 0,
                                fontSize: '1rem',
                                color: policyData.driscScore >= 8 ? '#059669' :
                                    policyData.driscScore >= 6 ? '#d97706' :
                                        '#dc2626',
                                fontWeight: '600'
                            }}>
                                {policyData.driscScore >= 80 ? 'Excellent' :
                                    policyData.driscScore >= 60 ? 'Good' :
                                        'Needs Attention'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default InsurancePage;