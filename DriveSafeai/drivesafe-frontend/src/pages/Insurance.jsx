import { Container, Row, Col, Card, Form, Button, Alert, Tab, Tabs } from 'react-bootstrap';
import axios from 'axios';
import { useState } from 'react';

const Insurance = () => {
  const [activeTab, setActiveTab] = useState('create');
  const [policyData, setPolicyData] = useState({
    policyNumber: '',
    userId: '',
    vehicleId: '',
    policyStartDate: '',
    policyEndDate: '',
    coverageType: '',
    coverageAmount: '',
    basePremium: ''
  });
  const [premiumUserId, setPremiumUserId] = useState('');
  const [premiumResult, setPremiumResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handlePolicyChange = (e) => {
    setPolicyData({
      ...policyData,
      [e.target.name]: e.target.value
    });
  };

  const handleCreatePolicy = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');

    try {
      const response = await axios.post('/api/insurance/policy', policyData);
      setMessage('Policy created successfully!');
      setPolicyData({
        policyNumber: '',
        userId: '',
        vehicleId: '',
        policyStartDate: '',
        policyEndDate: '',
        coverageType: '',
        coverageAmount: '',
        basePremium: ''
      });
    } catch (err) {
      setError('Failed to create policy');
    } finally {
      setLoading(false);
    }
  };

  const handleCalculatePremium = async () => {
    if (!premiumUserId) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`/api/insurance/premium/${premiumUserId}`);
      setPremiumResult(response.data);
    } catch (err) {
      setError('Failed to calculate premium');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="mt-5">
      <Row>
        <Col>
          <h2>Insurance Management</h2>
        </Col>
      </Row>
      
      <Row>
        <Col>
          <Tabs activeKey={activeTab} onSelect={(k) => setActiveTab(k)} className="mb-3">
            <Tab eventKey="create" title="Create Policy">
              <Card>
                <Card.Header>
                  <h5>Create New Insurance Policy</h5>
                </Card.Header>
                <Card.Body>
                  {message && <Alert variant="success">{message}</Alert>}
                  {error && <Alert variant="danger">{error}</Alert>}
                  
                  <Form onSubmit={handleCreatePolicy}>
                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Policy Number</Form.Label>
                          <Form.Control
                            type="text"
                            name="policyNumber"
                            value={policyData.policyNumber}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>User ID</Form.Label>
                          <Form.Control
                            type="number"
                            name="userId"
                            value={policyData.userId}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Vehicle ID</Form.Label>
                          <Form.Control
                            type="number"
                            name="vehicleId"
                            value={policyData.vehicleId}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Coverage Type</Form.Label>
                          <Form.Select
                            name="coverageType"
                            value={policyData.coverageType}
                            onChange={handlePolicyChange}
                            required
                          >
                            <option value="">Select Coverage Type</option>
                            <option value="COMPREHENSIVE">Comprehensive</option>
                            <option value="THIRD_PARTY">Third Party</option>
                            <option value="COLLISION">Collision</option>
                          </Form.Select>
                        </Form.Group>
                      </Col>
                      
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Policy Start Date</Form.Label>
                          <Form.Control
                            type="date"
                            name="policyStartDate"
                            value={policyData.policyStartDate}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Policy End Date</Form.Label>
                          <Form.Control
                            type="date"
                            name="policyEndDate"
                            value={policyData.policyEndDate}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Coverage Amount ($)</Form.Label>
                          <Form.Control
                            type="number"
                            step="0.01"
                            name="coverageAmount"
                            value={policyData.coverageAmount}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Base Premium ($)</Form.Label>
                          <Form.Control
                            type="number"
                            step="0.01"
                            name="basePremium"
                            value={policyData.basePremium}
                            onChange={handlePolicyChange}
                            required
                          />
                        </Form.Group>
                      </Col>
                    </Row>
                    
                    <Button type="submit" variant="primary" disabled={loading} className="w-100">
                      {loading ? 'Creating Policy...' : 'Create Policy'}
                    </Button>
                  </Form>
                </Card.Body>
              </Card>
            </Tab>
            
            <Tab eventKey="premium" title="Calculate Premium">
              <Card>
                <Card.Header>
                  <h5>Premium Calculator</h5>
                </Card.Header>
                <Card.Body>
                  {error && <Alert variant="danger">{error}</Alert>}
                  
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label>User ID</Form.Label>
                        <div className="d-flex">
                          <Form.Control
                            type="number"
                            value={premiumUserId}
                            onChange={(e) => setPremiumUserId(e.target.value)}
                            placeholder="Enter User ID"
                          />
                          <Button 
                            variant="primary" 
                            className="ms-2"
                            onClick={handleCalculatePremium}
                            disabled={!premiumUserId || loading}
                          >
                            Calculate
                          </Button>
                        </div>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  {premiumResult && (
                    <Card className="mt-4">
                      <Card.Header>
                        <h6>Premium Calculation Result</h6>
                      </Card.Header>
                      <Card.Body>
                        <Row>
                          <Col md={6}>
                            <p><strong>Policy ID:</strong> {premiumResult.policyId}</p>
                            <p><strong>Risk Score:</strong> {premiumResult.riskScore?.toFixed(2)}</p>
                            <p><strong>Risk Category:</strong> {premiumResult.riskCategory}</p>
                          </Col>
                          <Col md={6}>
                            <p><strong>Base Premium:</strong> ${premiumResult.basePremium}</p>
                            <p><strong>Risk Multiplier:</strong> {premiumResult.riskMultiplier}x</p>
                            <h5><strong>Final Premium:</strong> ${premiumResult.finalPremium}</h5>
                          </Col>
                        </Row>
                      </Card.Body>
                    </Card>
                  )}
                </Card.Body>
              </Card>
            </Tab>
          </Tabs>
        </Col>
      </Row>
    </Container>
  );
};

export default Insurance;
