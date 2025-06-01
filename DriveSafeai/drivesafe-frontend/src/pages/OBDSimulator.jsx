import React, { useState, useEffect, useRef } from 'react';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import TripService from "../services/TripService.js";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
    BarElement
);

const OBDDashboard = () => {
  const [obdData, setObdData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [fetchCount, setFetchCount] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [sessionId, setSessionId] = useState(null);
  const [tripStatus, setTripStatus] = useState('no-trip'); // 'no-trip', 'active', 'completed'
  const [lastDataHash, setLastDataHash] = useState(null);
  const [unchangedDataCount, setUnchangedDataCount] = useState(0);
  const [isOffline, setIsOffline] = useState(false);

  const intervalRef = useRef(null);
  const sessionTimeoutRef = useRef(null);

  const [historicalData, setHistoricalData] = useState({
    rpm: [],
    speed: [],
    throttle: [],
    temperature: [],
    timestamps: []
  });

  const JSON_URL = 'http://192.0.0.2:9999';
  const SESSION_STORAGE_KEY = 'session';

  // Session management functions
  const getStoredSession = () => {
    try {
      const stored = localStorage.getItem(SESSION_STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.error('Error reading session from localStorage:', error);
      return null;
    }
  };

  const storeSession = (sessionData) => {
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(sessionData));
      console.log('Session stored in localStorage:', sessionData);
    } catch (error) {
      console.error('Error storing session in localStorage:', error);
    }
  };

  const clearStoredSession = () => {
    try {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      console.log('Session cleared from localStorage');
    } catch (error) {
      console.error('Error clearing session from localStorage:', error);
    }
  };

  // Generate session ID
  const generateSessionId = () => {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  };

  // Initialize or get session
  const initializeSession = () => {
    // First check if there's a stored session
    const storedSession = getStoredSession();

    if (storedSession && storedSession.id && storedSession.status === 'active') {
      console.log('Found existing active session in localStorage:', storedSession.id);
      setSessionId(storedSession.id);
      setTripStatus('active');
      return storedSession.id;
    } else {
      // Generate new session
      const newSessionId = generateSessionId();
      const sessionData = {
        id: newSessionId,
        status: 'active',
        createdAt: new Date().toISOString(),
        lastActivity: new Date().toISOString()
      };

      storeSession(sessionData);
      setSessionId(newSessionId);
      console.log('New session created and stored:', newSessionId);
      return newSessionId;
    }
  };

  // Update session activity
  const updateSessionActivity = (currentSessionId) => {
    if (currentSessionId) {
      const sessionData = {
        id: currentSessionId,
        status: 'active',
        createdAt: getStoredSession()?.createdAt || new Date().toISOString(),
        lastActivity: new Date().toISOString()
      };
      storeSession(sessionData);
    }
  };

  // Create a hash of the critical data to detect changes
  const createDataHash = (data) => {
    if (!data) return null;
    return JSON.stringify({
      speed: data.vehicle_speed || data.speed,
      rpm: data.engine_rpm || data.rpm,
      throttlePosition: data.throttle_position || data.throttlePosition
    });
  };

  // Send data to TripService
  const sendToTripService = async (data) => {
    try {
      // Get or initialize session
      let currentSessionId = sessionId;
      if (!currentSessionId) {
        currentSessionId = initializeSession();
      }

      // Update session activity
      updateSessionActivity(currentSessionId);

      // Create trip data object matching your required format
      const tripData = {
        sessionId: currentSessionId,
        observationHour: parseFloat(data.observation_hour || 0),
        speed: parseFloat(data.speed || 0),
        rpm: parseFloat(data.rpm || 0),
        acceleration: parseFloat(data.acceleration || 0),
        throttlePosition: parseFloat(data.throttle_position || 0),
        engineTemperature: parseFloat(data.engine_temperature || 0),
        systemVoltage: parseFloat(data.system_voltage || 0),
        distanceTravelled: parseFloat(data.distance_travelled || 0),
        engineLoadValue: parseFloat(data.engine_load_value || 0),
        latitude: parseFloat(data.latitude || 0),
        longitude: parseFloat(data.longitude || 0),
        altitude: parseFloat(data.altitude || 0),
        idVehicle: parseInt(data.id_vehicle || 0),
        bodyTemperature: parseFloat(data.body_temperature || 0),
        idDriver: parseInt(data.id_driver || 0),
        currentWeather: parseFloat(data.current_weather || 0),
        hasPrecipitation: parseInt(data.has_precipitation || 0),
        isDayTime: parseInt(data.is_day_time || 0),
        temperature: parseFloat(data.temperature || 0),
        windSpeed: parseFloat(data.wind_speed || 0),
        windDirection: parseFloat(data.wind_direction || 0),
        relativeHumidity: parseFloat(data.relative_humidity || 0),
        visibility: parseFloat(data.visibility || 0),
        uvIndex: parseFloat(data.uv_index || 0),
        cloudCover: parseFloat(data.cloud_cover || 0),
        ceiling: parseFloat(data.ceiling || 0),
        pressure: parseFloat(data.pressure || 0),
        precipitation: parseFloat(data.precipitation || 0),
        accidentsOnsite: parseInt(data.accidents_onsite || 0),
        designSpeed: parseFloat(data.design_speed || 0),
        accidentsTime: parseInt(data.accidents_time || 0)
      };


      // Call TripService.live()
      const result = await TripService.live(tripData);
      console.log('TripService.live() result:', result);
      return result;
    } catch (error) {
      console.error('Error sending data to TripService:', error);
      throw error;
    }
  };

  // End session
  const endTripSession = async (currentSessionId = null) => {
    const sessionToEnd = currentSessionId || sessionId;
    const storedSession = getStoredSession();

    // Use session from localStorage if no session provided and no current session
    const finalSessionId = sessionToEnd || storedSession?.id;

    if (finalSessionId) {
      console.log('Attempting to end session:', finalSessionId);

      try {
        await TripService.endSession(finalSessionId);
        console.log('Session ended successfully:', finalSessionId);
      } catch (error) {
        console.error('Error ending session:', error);
        // Continue with state update even if API fails
      }

      // Update session status in localStorage before clearing
      if (storedSession) {
        const completedSessionData = {
          ...storedSession,
          status: 'completed',
          completedAt: new Date().toISOString()
        };
        storeSession(completedSessionData);

        // Clear after a short delay to allow for any final processing
        setTimeout(() => {
          clearStoredSession();
        }, 1000);
      } else {
        clearStoredSession();
      }

      // Always update state after attempting to end session
      setTripStatus('completed');
      setSessionId(null);
      setUnchangedDataCount(0);
      setLastDataHash(null);

      // Clear intervals
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (sessionTimeoutRef.current) {
        clearTimeout(sessionTimeoutRef.current);
        sessionTimeoutRef.current = null;
      }

      return true;
    }
    return false;
  };

  // Handle offline scenarios - improved function
  const handleOfflineState = async () => {
    console.log('Handling offline state...');

    const storedSession = getStoredSession();
    const activeSessionId = sessionId || storedSession?.id;

    console.log('Current sessionId:', sessionId);
    console.log('Stored session:', storedSession);
    console.log('Current tripStatus:', tripStatus);

    // Check if there's an active session (either in state or localStorage)
    if (activeSessionId && (storedSession?.status === 'active' || sessionId)) {
      console.log('Active session found, ending session due to offline state');

      try {
        // Call the endSession API
        await TripService.endSession(activeSessionId);
        console.log('Session ended successfully due to offline condition');
      } catch (error) {
        console.error('Failed to end session via API, but continuing with state cleanup:', error);
      }

      // Update session as completed in localStorage
      if (storedSession) {
        const completedSessionData = {
          ...storedSession,
          status: 'completed',
          completedAt: new Date().toISOString(),
          reason: 'offline'
        };
        storeSession(completedSessionData);

        // Clear after processing
        setTimeout(() => {
          clearStoredSession();
        }, 1000);
      } else {
        clearStoredSession();
      }

      // Clear session state
      setSessionId(null);
      setTripStatus('completed');
      setUnchangedDataCount(0);
      setLastDataHash(null);

      // Clear any running intervals
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (sessionTimeoutRef.current) {
        clearTimeout(sessionTimeoutRef.current);
        sessionTimeoutRef.current = null;
      }
    }

    // Set offline state
    setIsOffline(true);
    setConnectionStatus('offline');
    setError(null);
  };

  const fetchOBDData = async () => {
    try {
      setConnectionStatus('connecting');

      const response = await fetch(JSON_URL, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        cache: 'no-cache'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Handle both data structures - direct data or nested data.data
      const actualData = data.data || data;

      // Clear offline state and error when connection is successful
      setIsOffline(false);
      setError(null);

      // Check if we have valid trip data
      if (!actualData) {
        setTripStatus('no-trip');
        setConnectionStatus('no-data');
        setLoading(false);
        return;
      }

      setObdData(data);
      setLastUpdated(new Date().toLocaleTimeString());
      setFetchCount(prev => prev + 1);
      setConnectionStatus('connected');
      setLoading(false);

      // Get speed and rpm values (handle both data structures)
      const currentSpeed = actualData.vehicle_speed || actualData.speed || 0;
      const currentRpm = actualData.engine_rpm || actualData.rpm || 0;

      // Check if trip is active (speed > 0 or rpm > 0)
      if (currentSpeed > 0 || currentRpm > 0) {
        if (tripStatus === 'no-trip' || tripStatus === 'completed') {
          setTripStatus('active');
          setUnchangedDataCount(0);
        }

        // Create hash of current data
        const currentHash = createDataHash(actualData);

        // Check if data has changed
        if (currentHash === lastDataHash && currentSpeed === 0) {
          setUnchangedDataCount(prev => prev + 1);

          // If data hasn't changed for 6 cycles (30 seconds with 5-second intervals) and speed is 0
          if (unchangedDataCount >= 5) {
            console.log('Data unchanged for 30 seconds with speed = 0, ending session');
            await endTripSession();
            return;
          }
        } else {
          setUnchangedDataCount(0);
        }

        setLastDataHash(currentHash);

        // Update historical data for charts
        const now = new Date().toLocaleTimeString();
        setHistoricalData(prev => {
          const newData = {
            rpm: [...prev.rpm.slice(-19), currentRpm],
            speed: [...prev.speed.slice(-19), currentSpeed],
            throttle: [...prev.throttle.slice(-19), actualData.throttlePosition || actualData.throttlePosition || 0],
            temperature: [...prev.temperature.slice(-19), actualData.coolant_temperature || actualData.engineTemperature || 0],
            timestamps: [...prev.timestamps.slice(-19), now]
          };
          return newData;
        });

        // Send data to TripService
        await sendToTripService(actualData);
      } else if (tripStatus === 'active') {
        // Vehicle stopped, start monitoring for session end
        const currentHash = createDataHash(actualData);
        if (currentHash === lastDataHash) {
          setUnchangedDataCount(prev => prev + 1);

          if (unchangedDataCount >= 5) {
            console.log('Vehicle stopped for 30 seconds, ending session');
            await endTripSession();
          }
        }
        setLastDataHash(currentHash);
      }

    } catch (err) {
      console.error('Error fetching OBD data:', err);
      setLoading(false);

      // Handle offline state with improved logic
      await handleOfflineState();
    }
  };

  useEffect(() => {
    // Check for existing session on component mount
    const storedSession = getStoredSession();
    if (storedSession && storedSession.status === 'active') {
      console.log('Restoring session from localStorage:', storedSession.id);
      setSessionId(storedSession.id);
      setTripStatus('active');
    }

    // Initial fetch
    fetchOBDData();

    // Set up interval
    intervalRef.current = setInterval(fetchOBDData, 5000); // Fetch every 5 seconds

    return () => {
      // Cleanup
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (sessionTimeoutRef.current) {
        clearTimeout(sessionTimeoutRef.current);
      }
      // End session if active
      const currentSessionId = sessionId || getStoredSession()?.id;
      if (currentSessionId) {
        console.log('Component unmounting, ending active session');
        endTripSession(currentSessionId).catch(console.error);
      }
    };
  }, []);

  // Chart configurations
  const rpmChartData = {
    labels: historicalData.timestamps,
    datasets: [{
      label: 'Engine RPM',
      data: historicalData.rpm,
      borderColor: 'rgb(255, 99, 132)',
      backgroundColor: 'rgba(255, 99, 132, 0.2)',
      tension: 0.4,
      fill: true
    }]
  };

  const speedChartData = {
    labels: historicalData.timestamps,
    datasets: [{
      label: 'Vehicle Speed',
      data: historicalData.speed,
      borderColor: 'rgb(54, 162, 235)',
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      tension: 0.4,
      fill: true
    }]
  };

  const fuelGaugeData = {
    labels: ['Fuel Level', 'Empty'],
    datasets: [{
      data: [
        (obdData?.data?.fuel_level || obdData?.fuel_level || 0),
        100 - (obdData?.data?.fuel_level || obdData?.fuel_level || 0)
      ],
      backgroundColor: ['#28a745', '#e9ecef'],
      borderWidth: 0
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        display: false
      },
      y: {
        beginAtZero: true
      }
    }
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success';
      case 'connecting': return 'warning';
      case 'error': return 'danger';
      case 'offline': return 'secondary';
      case 'no-data': return 'secondary';
      default: return 'secondary';
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return '🟢';
      case 'connecting': return '🟡';
      case 'error': return '🔴';
      case 'offline': return '⚫';
      case 'no-data': return '⚫';
      default: return '⚫';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'danger';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'secondary';
    }
  };

  // Get data values with fallback for both data structures
  const getData = (key1, key2) => {
    return obdData?.data?.[key1] || obdData?.[key2] || obdData?.[key1] || 0;
  };

  return (
      <div className="container-fluid py-4 bg-dark text-light min-vh-100">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet" />
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet" />

        <style jsx>{`
          .metric-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border: none;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
          }
          .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
          }
          .pulse {
            animation: pulse 2s infinite;
          }
          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
          }
          .chart-container {
            height: 200px;
          }
          .gauge-container {
            height: 150px;
          }
        `}</style>

        {/* Header */}
        <div className="row mb-4">
          <div className="col-12">
            <div className="card bg-primary text-white">
              <div className="card-body">
                <div className="row align-items-center">
                  <div className="col-md-6">
                    <h1 className="h3 mb-0">
                      <i className="fas fa-car me-3"></i>
                      OBD-II Live Dashboard
                    </h1>
                    {obdData?.vehicle && (
                        <p className="mb-0 mt-2">
                          {obdData.vehicle.year} {obdData.vehicle.make} {obdData.vehicle.model}
                        </p>
                    )}
                    {sessionId && (
                        <p className="mb-0 mt-1">
                          <small><strong>Session ID:</strong> {sessionId}</small>
                        </p>
                    )}
                  </div>
                  <div className="col-md-6 text-md-end">
                    <div className="d-flex justify-content-md-end align-items-center">
                    <span className={`badge bg-${getStatusColor()} me-3 fs-6`}>
                      {getStatusIcon()} {connectionStatus.toUpperCase()}
                    </span>
                      <div className="text-end">
                        <small className="d-block">Last Update: {lastUpdated || 'Never'}</small>
                        <small className="d-block">Fetch Count: {fetchCount}</small>
                        {tripStatus === 'active' && sessionId && (
                            <small className="d-block text-success">
                              <i className="fas fa-play-circle me-1"></i>Trip Active
                            </small>
                        )}
                        {tripStatus === 'completed' && !isOffline && (
                            <small className="d-block text-info">
                              <i className="fas fa-stop-circle me-1"></i>Trip Completed
                            </small>
                        )}
                        {tripStatus === 'completed' && isOffline && (
                            <small className="d-block text-warning">
                              <i className="fas fa-exclamation-circle me-1"></i>Trip Completed - Disconnected
                            </small>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error Alert - Only show if there's an actual error to display */}
        {error && (
            <div className="alert alert-danger alert-dismissible mb-4">
              <i className="fas fa-exclamation-triangle me-2"></i>
              <strong>Connection Error:</strong> {error}
              <button type="button" className="btn-close" onClick={() => setError(null)}></button>
            </div>
        )}

        {/* No Active Trip Message - Show when offline and no session and no completed trip */}
        {isOffline && !sessionId && tripStatus === 'no-trip' && (
            <div className="alert alert-info mb-4">
              <i className="fas fa-info-circle me-2"></i>
              <strong>No Live trip currently active</strong>
            </div>
        )}

        {/* Trip Completed Message - Show when trip was completed */}
        {tripStatus === 'completed' && (
            <div className="alert alert-success mb-4">
              <i className="fas fa-check-circle me-2"></i>
              <strong>Trip Completed</strong>
              {isOffline ? ' - Device disconnected' : ''}
            </div>
        )}

        {loading && (
            <div className="text-center py-5">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
              <p className="mt-3">Connecting to OBD-II interface...</p>
            </div>
        )}

        {!loading && obdData && (
            <>
              {/* Main Metrics */}
              <div className="row mb-4">
                <div className="col-md-3 mb-3">
                  <div className="card metric-card text-white h-100">
                    <div className="card-body text-center">
                      <i className="fas fa-tachometer-alt fa-2x mb-3 text-warning"></i>
                      <h5 className="card-title">Speed</h5>
                      <h2 className="display-4 mb-0">{getData('speed', 'vehicle_speed')}</h2>
                      <small>mph</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-3 mb-3">
                  <div className="card metric-card text-white h-100">
                    <div className="card-body text-center">
                      <i className="fas fa-cog fa-2x mb-3 text-danger"></i>
                      <h5 className="card-title">Engine RPM</h5>
                      <h2 className="display-4 mb-0">{getData('rpm', 'engine_rpm')}</h2>
                      <small>rpm</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-3 mb-3">
                  <div className="card metric-card text-white h-100">
                    <div className="card-body text-center">
                      <i className="fas fa-route fa-2x mb-3 text-info"></i>
                      <h5 className="card-title">Distance</h5>
                      <h2 className="display-4 mb-0">{getData('distance_travelled', 'distance')}</h2>
                      <small>km</small>
                    </div>
                  </div>
                </div>
                <div className="col-md-3 mb-3">
                  <div className="card metric-card text-white h-100">
                    <div className="card-body text-center">
                      <i className="fas fa-tachometer-alt fa-2x mb-3 text-success"></i>
                      <h5 className="card-title">Acceleration</h5>
                      <h2 className="display-4 mb-0">{getData('acceleration', 'fuelLevel')}</h2>
                      <small>m/s²</small>
                    </div>
                  </div>
                </div>
              </div>

              {/* Charts */}
              <div className="row mb-4">
                <div className="col-md-6 mb-3">
                  <div className="card bg-secondary text-white">
                    <div className="card-header">
                      <h5 className="mb-0"><i className="fas fa-chart-line me-2"></i>Engine RPM Trend</h5>
                    </div>
                    <div className="card-body">
                      <div className="chart-container">
                        <Line data={rpmChartData} options={chartOptions} />
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-md-6 mb-3">
                  <div className="card bg-secondary text-white">
                    <div className="card-header">
                      <h5 className="mb-0"><i className="fas fa-chart-line me-2"></i>Speed Trend</h5>
                    </div>
                    <div className="card-body">
                      <div className="chart-container">
                        <Line data={speedChartData} options={chartOptions} />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Additional Metrics */}
              <div className="row mb-4">
                <div className="col-md-4 mb-3">
                  <div className="card bg-info text-white">
                    <div className="card-body">
                      <div className="row align-items-center">
                        <div className="col-4">
                          <i className="fas fa-battery-half fa-3x"></i>
                        </div>
                        <div className="col-8 text-end">
                          <h5 className="mb-0">Battery Voltage</h5>
                          <h3>{getData('system_voltage', 'systemVoltage')} V</h3>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-md-4 mb-3">
                  <div className="card bg-warning text-dark">
                    <div className="card-body">
                      <div className="row align-items-center">
                        <div className="col-4">
                          <i className="fas fa-weight fa-3x"></i>
                        </div>
                        <div className="col-8 text-end">
                          <h5 className="mb-0">Engine Load</h5>
                          <h3>{getData('engine_load_value', 'engineLoadValue')}%</h3>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-md-4 mb-3">
                  <div className="card bg-success text-white">
                    <div className="card-body">
                      <div className="row align-items-center">
                        <div className="col-4">
                          <i className="fas fa-compress-arrows-alt fa-3x"></i>
                        </div>
                        <div className="col-8 text-end">
                          <h5 className="mb-0">Throttle Position</h5>
                          <h3>{getData('throttle_position', 'throttlePosition')}%</h3>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Fuel Gauge */}
              <div className="row mb-4">
                <div className="col-md-6 mx-auto">
                  <div className="card bg-secondary text-white">
                    <div className="card-header text-center">
                      <h5 className="mb-0"><i className="fas fa-gas-pump me-2"></i>Fuel Level</h5>
                    </div>
                    <div className="card-body">
                      <div className="gauge-container">
                        <Doughnut
                            data={fuelGaugeData}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              circumference: 180,
                              rotation: 270,
                              plugins: {
                                legend: {
                                  display: false
                                }
                              }
                            }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* DTC Codes */}
              {obdData?.dtc_codes && obdData.dtc_codes.length > 0 && (
                  <div className="row mb-4">
                    <div className="col-12">
                      <div className="card bg-danger text-white">
                        <div className="card-header">
                          <h5 className="mb-0">
                            <i className="fas fa-exclamation-triangle me-2"></i>
                            Diagnostic Trouble Codes ({obdData.dtc_codes.length})
                          </h5>
                        </div>
                        <div className="card-body">
                          <div className="row">
                            {obdData.dtc_codes.map((code, index) => (
                                <div key={index} className="col-md-6 mb-2">
                                  <div className={`alert alert-${getSeverityColor(code.severity)} mb-2`}>
                                    <strong>{code.code}</strong>: {code.description}
                                    <br />
                                    <small>Severity: {code.severity}</small>
                                  </div>
                                </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
              )}

              {/* Vehicle Information */}
              {obdData?.vehicle && (
                  <div className="row">
                    <div className="col-12">
                      <div className="card bg-secondary text-white">
                        <div className="card-header">
                          <h5 className="mb-0">
                            <i className="fas fa-info-circle me-2"></i>Vehicle Information
                          </h5>
                        </div>
                        <div className="card-body">
                          <div className="row">
                            <div className="col-md-3">
                              <strong>Make:</strong> {obdData.vehicle.make}
                            </div>
                            <div className="col-md-3">
                              <strong>Model:</strong> {obdData.vehicle.model}
                            </div>
                            <div className="col-md-3">
                              <strong>Year:</strong> {obdData.vehicle.year}
                            </div>
                            <div className="col-md-3">
                              <strong>VIN:</strong> {obdData.vehicle.vin}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
              )}
            </>
        )}
      </div>
  );
};

export default OBDDashboard;