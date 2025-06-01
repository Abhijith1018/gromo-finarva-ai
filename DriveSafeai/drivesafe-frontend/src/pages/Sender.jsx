import React, { useState, useEffect, useCallback } from 'react';
import { Radio, Send, AlertCircle, Play, Pause } from 'lucide-react';

const BlePeripheralPage = () => {
  const [isAdvertising, setIsAdvertising] = useState(false);
  const [transmissionData, setTransmissionData] = useState({
    temperature: 25,
    humidity: 60,
    pressure: 1013,
    timestamp: Date.now()
  });
  const [customData, setCustomData] = useState('');
  const [transmissionInterval, setTransmissionInterval] = useState(2000);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('Ready to advertise');
  const [sentCount, setSentCount] = useState(0);

  // Check if Web Bluetooth is supported (needed for peripheral mode)
  const isBluetoothSupported = 'bluetooth' in navigator;

  // Simulate sensor data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTransmissionData(prev => ({
        temperature: Math.round((20 + Math.random() * 10) * 10) / 10,
        humidity: Math.round((50 + Math.random() * 30) * 10) / 10,
        pressure: Math.round((1000 + Math.random() * 50) * 10) / 10,
        timestamp: Date.now()
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Note: Web Bluetooth API doesn't support peripheral mode in browsers yet
  // This is a simulation of what the peripheral would do
  const startAdvertising = useCallback(async () => {
    if (!isBluetoothSupported) {
      setError('Web Bluetooth is not supported in this browser/OS');
      return;
    }

    // In a real implementation, this would use Bluetooth peripheral APIs
    // For now, we'll simulate the advertising process
    setError('Note: Web browsers don\'t support BLE peripheral mode yet. This is a simulation.');
    setStatus('Simulating BLE advertising...');
    setIsAdvertising(true);
    setSentCount(0);
  }, []);

  const stopAdvertising = useCallback(() => {
    setIsAdvertising(false);
    setStatus('Stopped advertising');
  }, []);

  // Simulate data transmission
  useEffect(() => {
    if (!isAdvertising) return;

    const interval = setInterval(() => {
      const dataToSend = customData ? 
        JSON.parse(customData) : 
        transmissionData;
      
      // Simulate sending data
      console.log('Simulating BLE transmission:', dataToSend);
      setSentCount(prev => prev + 1);
      setStatus(`Broadcasting data... (${sentCount + 1} packets sent)`);
    }, transmissionInterval);

    return () => clearInterval(interval);
  }, [isAdvertising, transmissionData, customData, transmissionInterval, sentCount]);

  const handleCustomDataChange = (e) => {
    setCustomData(e.target.value);
  };

  const sendManualData = () => {
    if (!isAdvertising) {
      setError('Start advertising first');
      return;
    }

    try {
      const dataToSend = customData ? JSON.parse(customData) : transmissionData;
      console.log('Manual BLE transmission:', dataToSend);
      setSentCount(prev => prev + 1);
      setStatus(`Manual data sent! (${sentCount + 1} packets total)`);
      setError('');
    } catch (err) {
      setError('Invalid JSON in custom data');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center gap-3 mb-6">
            <Radio className="h-8 w-8 text-green-600" />
            <h1 className="text-2xl font-bold text-gray-800">BLE Peripheral Device</h1>
          </div>

          {/* Important Notice */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-yellow-600" />
              <span className="font-medium text-yellow-800">Important Note:</span>
            </div>
            <p className="text-yellow-700 mt-1 text-sm">
              Web browsers don't support BLE peripheral mode yet. This page simulates the peripheral functionality. 
              For real BLE peripheral, you'd need a native app or dedicated hardware.
            </p>
          </div>

          {/* Status Section */}
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 mb-2">
              {isAdvertising ? (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="font-medium">Broadcasting</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                  <span className="font-medium">Status</span>
                </div>
              )}
            </div>
            <p className="text-gray-700">{status}</p>
            {isAdvertising && (
              <p className="text-sm text-gray-500 mt-1">
                Transmitting every {transmissionInterval / 1000}s
              </p>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-red-600" />
                <span className="text-red-800 font-medium">Notice:</span>
              </div>
              <p className="text-red-700 mt-1 text-sm">{error}</p>
            </div>
          )}

          {/* Control Section */}
          <div className="space-y-4 mb-6">
            <div className="flex gap-3">
              {!isAdvertising ? (
                <button
                  onClick={startAdvertising}
                  className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <Play className="h-4 w-4" />
                  Start Advertising
                </button>
              ) : (
                <button
                  onClick={stopAdvertising}
                  className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <Pause className="h-4 w-4" />
                  Stop Advertising
                </button>
              )}
              
              <button
                onClick={sendManualData}
                disabled={!isAdvertising}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                <Send className="h-4 w-4" />
                Send Now
              </button>
            </div>

            {/* Transmission Interval */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Transmission Interval (ms)
              </label>
              <input
                type="number"
                value={transmissionInterval}
                onChange={(e) => setTransmissionInterval(parseInt(e.target.value) || 1000)}
                min="500"
                max="10000"
                step="500"
                className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Current Sensor Data */}
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">Current Sensor Data</h2>
            <div className="bg-gray-50 rounded-lg p-4">
              <pre className="text-sm bg-white p-3 rounded border overflow-x-auto">
                {JSON.stringify(transmissionData, null, 2)}
              </pre>
            </div>
          </div>

          {/* Custom Data Section */}
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">Custom Data (Optional)</h2>
            <p className="text-sm text-gray-600 mb-3">
              Enter custom JSON data to transmit instead of sensor data:
            </p>
            <textarea
              value={customData}
              onChange={handleCustomDataChange}
              placeholder='{"message": "Hello from BLE!", "value": 42}'
              className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
            />
          </div>

          {/* Statistics */}
          <div className="border-t pt-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">Statistics</h2>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-600">{sentCount}</div>
                <div className="text-sm text-gray-600">Packets Sent</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-600">
                  {isAdvertising ? 'ON' : 'OFF'}
                </div>
                <div className="text-sm text-gray-600">Broadcasting</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-2xl font-bold text-purple-600">
                  {transmissionInterval / 1000}s
                </div>
                <div className="text-sm text-gray-600">Interval</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BlePeripheralPage;