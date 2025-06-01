import { useState } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert } from 'react-bootstrap';
import { useAuth } from '../utils/AuthContext';
import { tripAPI } from '../services/apiService';

const TripSubmission = () => {
  const [tripData, setTripData] = useState({
    speed: '',
    rpm: '',
    acceleration: '',
    throttlePosition: '',
    engineTemperature: '',
    systemVoltage: '',
    engineLoadValue: '',
    distanceTravelled: '',
    brake: '',
    vehicleId: ''
  });
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { user } = useAuth();

  const handleChange = (e) => {
    setTripData({
      ...tripData,
      [e.target.name]: parseFloat(e.target.value) || e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResponse(null);

    try {
      const response = await tripAPI.submitTrip(tripData);
      setResponse(response.data);
      
      // Reset form after successful submission
      setTripData({
        speed: '',
        rpm: '',
        acceleration: '',
        throttlePosition: '',
        engineTemperature: '',
        systemVoltage: '',
        engineLoadValue: '',
        distanceTravelled: '',
        brake: '',
        vehicleId: ''
      });
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to submit trip data');
    } finally {
      setLoading(false);
    }
  };

  const getDriveScoreColor = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'danger';
  };

  return (
    <Container className="mt-5">
      <Row>
        <Col>
          <h2>Submit Trip Data</h2>
          <p className="text-muted">Record your driving data for AI analysis</p>
        </Col>
      </Row>
      
      <Row>
        <Col md={8}>
          <Card>
            <Card.Header>
              <h5>Vehicle Trip Information</h5>
            </Card.Header>
            <Card.Body>
              {error && <Alert variant="danger">{error}</Alert>}
              
              <Form onSubmit={handleSubmit}>
                <Row>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Vehicle ID</Form.Label>
                      <Form.Control
                        type="number"
                        name="vehicleId"
                        value={tripData.vehicleId}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Speed (km/h)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="speed"
                        value={tripData.speed}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>RPM</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="rpm"
                        value={tripData.rpm}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Acceleration (m/s²)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="acceleration"
                        value={tripData.acceleration}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Throttle Position (%)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="throttlePosition"
                        value={tripData.throttlePosition}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                  </Col>
                  
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Engine Temperature (°C)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="engineTemperature"
                        value={tripData.engineTemperature}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>System Voltage (V)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="systemVoltage"
                        value={tripData.systemVoltage}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Engine Load Value (%)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="engineLoadValue"
                        value={tripData.engineLoadValue}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Distance Travelled (km)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="distanceTravelled"
                        value={tripData.distanceTravelled}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Brake Usage (%)</Form.Label>
                      <Form.Control
                        type="number"
                        step="0.1"
                        name="brake"
                        value={tripData.brake}
                        onChange={handleChange}
                        required
                      />
                    </Form.Group>
                  </Col>
                </Row>
                
                <Button type="submit" variant="primary" disabled={loading} className="w-100">
                  {loading ? 'Analyzing Trip...' : 'Submit Trip Data'}
                </Button>
              </Form>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={4}>
          {response && (
            <Card>
              <Card.Header>
                <h5>Trip Analysis Result</h5>
              </Card.Header>
              <Card.Body>
                <div className="text-center mb-3">
                  <h3 className={`text-${getDriveScoreColor(response.driveScore)}`}>
                    {response.driveScore?.toFixed(1)}/100
                  </h3>
                  <p className="text-muted">Drive Score</p>
                </div>
                <p><strong>Trip ID:</strong> {response.tripId}</p>
                <hr />
                <h6>Feedback:</h6>
                <p className="text-muted">{response.feedback}</p>
                
                <Button 
                  variant="outline-primary" 
                  size="sm" 
                  className="w-100 mt-3"
                  onClick={() => window.location.href = '/dashboard'}
                >
                  View Dashboard
                </Button>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default TripSubmission;
