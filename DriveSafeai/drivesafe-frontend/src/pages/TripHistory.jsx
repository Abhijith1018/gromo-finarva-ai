import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Table, Badge, Alert } from 'react-bootstrap';
import { useAuth } from '../utils/AuthContext';
import { tripAPI } from '../services/apiService';

const TripHistory = () => {
  const [trips, setTrips] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const { user } = useAuth();

  useEffect(() => {
    if (user?.userId) {
      fetchTrips();
    }
  }, [user]);

  const fetchTrips = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await tripAPI.getUserTrips(user.userId);
      setTrips(response.data);
    } catch (err) {
      setError('Failed to fetch trip history');
    } finally {
      setLoading(false);
    }
  };

  const getScoreBadgeVariant = (score) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'danger';
  };

  if (loading) {
    return (
      <Container className="mt-5">
        <div className="text-center">
          <div className="spinner-border" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      </Container>
    );
  }

  return (
    <Container className="mt-5">
      <Row>
        <Col>
          <h2>Trip History</h2>
          <p className="text-muted">Your driving performance over time</p>
        </Col>
      </Row>

      {error && <Alert variant="danger">{error}</Alert>}
      
      <Row>
        <Col>
          <Card>
            <Card.Header>
              <h5>All Trips</h5>
            </Card.Header>
            <Card.Body>
              {trips.length > 0 ? (
                <Table striped bordered hover responsive>
                  <thead>
                    <tr>
                      <th>Trip ID</th>
                      <th>Date</th>
                      <th>Distance (km)</th>
                      <th>Drive Score</th>
                      <th>Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trips.map((trip) => (
                      <tr key={trip.tripId}>
                        <td>{trip.tripId}</td>
                        <td>{new Date(trip.createdAt).toLocaleDateString()}</td>
                        <td>{trip.distanceTravelled}</td>
                        <td>
                          <Badge bg={getScoreBadgeVariant(trip.driveScore)}>
                            {trip.driveScore?.toFixed(1)}
                          </Badge>
                        </td>
                        <td>{trip.feedback}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              ) : (
                <Alert variant="info">
                  No trips found. <a href="/trip">Submit your first trip</a> to see your driving history.
                </Alert>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default TripHistory;
