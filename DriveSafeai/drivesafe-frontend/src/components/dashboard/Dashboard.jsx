import React, { useState, useEffect } from 'react';
import UserService from '../../services/UserService';
import {useNavigate} from "react-router-dom";

const Dashboard = () => {
  const [userData, setUserData] = useState(null);
  const [driscScore, setDriscScore] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scoreLoading, setScoreLoading] = useState(false);
  const [numberOfTrips, setNumberOfTrips] = useState(1);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const currentUser = localStorage.getItem('userId');
        if (currentUser) {
          const [user, score] = await Promise.all([
            UserService.getUserById(currentUser),
            UserService.getDriscScore(currentUser, numberOfTrips)
          ]);
          setUserData(user);
          setDriscScore(score);
          console.log('DRISC Score data:', score); // Debug log to check data structure
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [numberOfTrips]);

  const handleTripsChange = async (newN) => {
    setNumberOfTrips(newN);
    setScoreLoading(true);

    try {
      const currentUser = localStorage.getItem('userId');
      if (currentUser) {
        const score = await UserService.getDriscScore(currentUser, newN);
        setDriscScore(score);
        console.log('Updated DRISC Score data:', score);
      }
    } catch (error) {
      console.error('Error fetching updated score:', error);
    } finally {
      setScoreLoading(false);
    }
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return { level: 'Low', color: '#10b981', symbol: '✓' };
    if (score >= 60) return { level: 'Medium', color: '#f59e0b', symbol: '⚠' };
    return { level: 'High', color: '#ef4444', symbol: '⚠' };
  };

  const getScoreColor = (score) => {
    if (score >= 80) return '#10b981';
    if (score >= 60) return '#f59e0b';
    return '#ef4444';
  };

  const handleStartTrip = () => {
    navigate("/trip-monitor");
  };

  if (loading) {
    return (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              width: '50px',
              height: '50px',
              border: '4px solid rgba(255,255,255,0.3)',
              borderTop: '4px solid white',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 20px'
            }}></div>
            <p style={{ color: 'white', fontSize: '16px', margin: 0 }}>Loading dashboard...</p>
            <style>
              {`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}
            </style>
          </div>
        </div>
    );
  }

  const riskData = driscScore ? getRiskLevel(driscScore.score) : null;
  const tripsCount = driscScore?.tripsConsidered || driscScore?.trips_considered || 0;

  const cardStyle = {
    background: 'white',
    borderRadius: '16px',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
    padding: '24px',
    transition: 'all 0.3s ease',
    border: '1px solid rgba(0, 0, 0, 0.05)',
    cursor: 'default'
  };

  const statCardStyle = {
    ...cardStyle,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    minHeight: '140px',
    position: 'relative',
    overflow: 'hidden'
  };

  const buttonStyle = {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    padding: '12px 24px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
  };

  const badgeStyle = (color) => ({
    background: color,
    color: 'white',
    padding: '4px 12px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: '600',
    border: 'none'
  });

  const progressBarStyle = {
    width: '100%',
    height: '8px',
    backgroundColor: '#f3f4f6',
    borderRadius: '4px',
    overflow: 'hidden',
    marginBottom: '12px'
  };

  const progressFillStyle = (score, color) => ({
    height: '100%',
    backgroundColor: color,
    width: `${score}%`,
    borderRadius: '4px',
    transition: 'width 1s ease-in-out'
  });

  const selectStyle = {
    background: 'white',
    border: '2px solid #e5e7eb',
    borderRadius: '8px',
    padding: '6px 12px',
    fontSize: '14px',
    fontWeight: '600',
    color: '#374151',
    cursor: 'pointer',
    outline: 'none',
    transition: 'all 0.3s ease'
  };

  return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
      }}>
        {/* Header */}
        <header style={{
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(0, 0, 0, 0.08)',
          padding: '20px 0',
          position: 'sticky',
          top: 0,
          zIndex: 100
        }}>
          <div style={{
            maxWidth: '1200px',
            margin: '0 auto',
            padding: '0 20px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: '700', color: '#1f2937' }}>
                Risk Scoring Dashboard
              </h1>
              <p style={{ margin: 0, color: '#6b7280', fontSize: '16px' }}>
                Monitor and analyze driving risk metrics
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <button
                  style={buttonStyle}
                  onClick={handleStartTrip}
                  onMouseEnter={(e) => e.target.style.transform = 'translateY(-2px)'}
                  onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
              >
                <span>🚗</span>
                Show Live Trip
              </button>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                background: 'rgba(0, 0, 0, 0.05)',
                borderRadius: '12px',
                padding: '12px 16px',
                gap: '8px'
              }}>
                <span>👤</span>
                <span style={{ fontWeight: '600', color: '#1f2937' }}>
                {userData?.fullName || 'Loading...'}
              </span>
              </div>
            </div>
          </div>
        </header>

        <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px 20px' }}>
          {/* Stats Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '24px',
            marginBottom: '32px'
          }}>
            {/* Risk Score Card */}
            <div
                style={{
                  ...statCardStyle,
                  minHeight: '180px', // Increased height to accommodate the selector
                  opacity: scoreLoading ? 0.7 : 1
                }}
                onMouseEnter={(e) => e.target.style.transform = 'translateY(-4px)'}
                onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <p style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '14px', fontWeight: '600' }}>
                    DRISC SCORE
                  </p>
                  <h3 style={{
                    margin: 0,
                    fontSize: '36px',
                    fontWeight: '700',
                    color: getScoreColor(driscScore?.score || 0)
                  }}>
                    {scoreLoading ? '...' : (driscScore?.score?.toFixed(1) || '0.0')}
                  </h3>
                </div>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: `${riskData?.color || '#9ca3af'}20`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px'
                }}>
                  📊
                </div>
              </div>

              {/* Trips Selector */}
              <div style={{ margin: '16px 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>Number of trips:</span>
                  <select
                      value={numberOfTrips}
                      onChange={(e) => handleTripsChange(parseInt(e.target.value))}
                      style={{
                        ...selectStyle,
                        borderColor: scoreLoading ? '#9ca3af' : '#e5e7eb'
                      }}
                      disabled={scoreLoading}
                  >
                    {[1, 2, 3, 4, 5, 10, 15, 20, 25, 30].map(num => (
                        <option key={num} value={num}>N = {num}</option>
                    ))}
                  </select>
                </div>
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
                  {scoreLoading ? 'Calculating...' : `Based on last ${numberOfTrips} trip${numberOfTrips !== 1 ? 's' : ''}`}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>📈</span>
                <small style={{ color: '#6b7280' }}>Risk assessment</small>
              </div>
            </div>

            {/* Risk Level Card */}
            <div
                style={statCardStyle}
                onMouseEnter={(e) => e.target.style.transform = 'translateY(-4px)'}
                onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <p style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '14px', fontWeight: '600' }}>
                    RISK LEVEL
                  </p>
                  <h3 style={{
                    margin: 0,
                    fontSize: '24px',
                    fontWeight: '700',
                    color: riskData?.color || '#9ca3af'
                  }}>
                    {riskData?.level || 'N/A'}
                  </h3>
                </div>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: `${riskData?.color || '#9ca3af'}20`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px'
                }}>
                  {riskData?.symbol || '🛡'}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '16px' }}>
                <span>🛡</span>
                <small style={{ color: '#6b7280' }}>Current status</small>
              </div>
            </div>

            {/* Trips Analyzed Card */}
            <div
                style={statCardStyle}
                onMouseEnter={(e) => e.target.style.transform = 'translateY(-4px)'}
                onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <p style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '14px', fontWeight: '600' }}>
                    TRIPS ANALYZED
                  </p>
                  <h3 style={{ margin: 0, fontSize: '36px', fontWeight: '700', color: '#3b82f6' }}>
                    {tripsCount}
                  </h3>
                </div>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: '#3b82f620',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px'
                }}>
                  🚗
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '16px' }}>
                <span>🕒</span>
                <small style={{ color: '#6b7280' }}>
                  {tripsCount > 0 ? 'Data points collected' : 'No trips recorded'}
                </small>
              </div>
            </div>

            {/* Premium Impact Card */}
            <div
                style={statCardStyle}
                onMouseEnter={(e) => e.target.style.transform = 'translateY(-4px)'}
                onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <p style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '14px', fontWeight: '600' }}>
                    PREMIUM IMPACT
                  </p>
                  <h3 style={{ margin: 0, fontSize: '24px', fontWeight: '700', color: '#10b981' }}>
                    {driscScore?.score ? `${(100 - driscScore.score)/8}% Saved` : '0% Saved'}
                  </h3>
                </div>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: '#10b98120',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '24px'
                }}>
                  💰
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '16px' }}>
                <span>📉</span>
                <small style={{ color: '#6b7280' }}>Estimated savings</small>
              </div>
            </div>
          </div>

          {/* Main Content Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(300px, 1fr) minmax(500px, 2fr)',
            gap: '32px',
            marginBottom: '32px'
          }}>
            {/* User Profile Card */}
            <div style={cardStyle}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '24px' }}>
                <div style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginRight: '16px',
                  fontSize: '24px'
                }}>
                  👤
                </div>
                <div>
                  <h5 style={{ margin: '0 0 4px 0', fontSize: '18px', fontWeight: '600', color: '#1f2937' }}>
                    {userData?.fullName || 'Loading...'}
                  </h5>
                  <p style={{ margin: 0, color: '#6b7280', fontSize: '14px' }}>
                    {userData?.email || 'Loading...'}
                  </p>
                </div>
              </div>

              <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '20px' }}>
                {[
                  { icon: '🚗', label: 'Vehicle', value: userData?.vehicleNo || 'Loading...' },
                  { icon: '📍', label: 'User ID', value: `#${userData?.userId || 'Loading...'}` },
                  { icon: '🛣️', label: 'Total Trips', value: `${tripsCount} trips` }
                ].map((item, index) => (
                    <div key={index} style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '12px 0',
                      borderTop: index > 0 ? '1px solid #f3f4f6' : 'none'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{item.icon}</span>
                        <span style={{ fontWeight: '500', color: '#374151' }}>{item.label}</span>
                      </div>
                      <span style={{ color: '#1f2937', fontWeight: '600' }}>{item.value}</span>
                    </div>
                ))}
              </div>
            </div>

            {/* Risk Analysis Card */}
            <div style={cardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <h5 style={{ margin: 0, fontSize: '20px', fontWeight: '600', color: '#1f2937' }}>
                  Risk Analysis
                </h5>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {[
                    { label: 'Safe', color: '#10b981' },
                    { label: 'Moderate', color: '#f59e0b' },
                    { label: 'High Risk', color: '#ef4444' }
                  ].map((badge, index) => (
                      <span key={index} style={badgeStyle(badge.color)}>
                    {badge.label}
                  </span>
                  ))}
                </div>
              </div>

              {/* Score Visualization */}
              <div style={{ marginBottom: '32px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <span style={{ fontWeight: '500', color: '#374151' }}>Overall Risk Score</span>
                  <span style={{
                    fontWeight: '700',
                    color: getScoreColor(driscScore?.score || 0),
                    fontSize: '16px'
                  }}>
                  {driscScore?.score?.toFixed(1) || '0.0'}/100
                </span>
                </div>
                <div style={progressBarStyle}>
                  <div style={progressFillStyle(driscScore?.score || 0, getScoreColor(driscScore?.score || 0))}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#6b7280' }}>
                  <span>Based on {numberOfTrips} trip{numberOfTrips !== 1 ? 's' : ''}</span>
                  <span>{numberOfTrips < 5 ? 'More trips needed for accuracy' : 'Sufficient data'}</span>
                </div>
              </div>

              {/* Risk Factors */}
              <div>
                <h6 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: '600', color: '#1f2937' }}>
                  Risk Factors
                </h6>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '12px' }}>
                  {[
                    { icon: '🏎', label: 'Speeding Events', level: 'Low', color: '#10b981' },
                    { icon: '🛑', label: 'Hard Braking', level: 'Moderate', color: '#f59e0b' },
                    { icon: '🔄', label: 'Sharp Turns', level: 'Low', color: '#10b981' },
                    { icon: '🌙', label: 'Night Driving', level: 'Normal', color: '#3b82f6' }
                  ].map((factor, index) => (
                      <div key={index} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '16px',
                        background: '#f8fafc',
                        borderRadius: '12px',
                        transition: 'all 0.3s ease'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span>{factor.icon}</span>
                          <span style={{ color: '#374151', fontWeight: '500' }}>{factor.label}</span>
                        </div>
                        <span style={badgeStyle(factor.color)}>{factor.level}</span>
                      </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div style={cardStyle}>
            <h5 style={{ margin: '0 0 24px 0', fontSize: '20px', fontWeight: '600', color: '#1f2937' }}>
              Recommendations
            </h5>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
              {[
                {
                  icon: numberOfTrips >= 5 ? "✅" : "ℹ️",
                  title: numberOfTrips >= 5 ? "Good Score!" : "Getting Started",
                  message: numberOfTrips >= 5
                      ? "Your driving behavior shows low risk patterns."
                      : `Try increasing N to ${Math.min(numberOfTrips + 2, 10)} for more comprehensive scoring.`,
                  color: numberOfTrips >= 5 ? '#3b82f6' : '#06b6d4'
                },
                {
                  icon: "⚠️",
                  title: "Data Analysis",
                  message: numberOfTrips < 10
                      ? "Consider analyzing more trips for better accuracy."
                      : "Your data is comprehensive for analysis.",
                  color: '#f59e0b'
                },
                {
                  icon: "💰",
                  title: "Premium Benefits",
                  message: numberOfTrips >= 5
                      ? "Eligible for reduced insurance premiums."
                      : "Analyze more trips to unlock premium benefits.",
                  color: '#10b981'
                }
              ].map((rec, index) => (
                  <div key={index} style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '16px',
                    padding: '20px',
                    background: `${rec.color}10`,
                    borderRadius: '12px',
                    border: `1px solid ${rec.color}20`
                  }}>
                    <span style={{ fontSize: '24px' }}>{rec.icon}</span>
                    <div>
                      <h6 style={{ margin: '0 0 8px 0', fontSize: '16px', fontWeight: '600', color: rec.color }}>
                        {rec.title}
                      </h6>
                      <p style={{ margin: 0, fontSize: '14px', color: '#374151', lineHeight: '1.5' }}>
                        {rec.message}
                      </p>
                    </div>
                  </div>
              ))}
            </div>
          </div>
        </main>
      </div>
  );
};

export default Dashboard;