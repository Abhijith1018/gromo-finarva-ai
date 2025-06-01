import { Container, Row, Col, Card, Form, Button, Alert, Tab, Tabs, Table } from 'react-bootstrap';
import axios from 'axios';
import { useState } from 'react';


const Claims = () => {
  const [activeTab, setActiveTab] = useState('file');
  const [claimData, setClaimData] = useState({
    policyId: '',
    claimNumber: '',
    claimDate: '',
    incidentDate: '',
    claimAmount: '',
    description: ''
  });
  const [policyId, setPolicyId] = useState('');
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleClaimChange = (e) => {
    setClaimData({
      ...claimData,
      [e.target.name]: e.target.value
    });
  };

  const handleFileClaim = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');

    try {
      const response = await axios.post('/api/insurance/claim', claimData);
      setMessage('Claim filed successfully!');
      setClaimData({
        policyId: '',
        claimNumber: '',
        claimDate: '',
        incidentDate: '',
        claimAmount: '',
        description: ''
      });
    } catch (err) {
      setError('Failed to file claim');
    } finally {
      setLoading(false);
    }
  };

  const handleGetClaims = async () => {
    if (!policyId) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`/api/insurance/claim/${policyId}`);
      setClaims(response.data);
    } catch (err) {
      setError('Failed to fetch claims');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="mt-5">
      <Row>
        <Col>
          <h2>Claims Management</h2>
        </Col>
      </Row>
      
      <Row>
        <Col>
          <Tabs activeKey={activeTab} onSelect={(k) => setActiveTab(k)} className="mb-3">
            <Tab eventKey="file" title="File New Claim">
              <Card>
                <Card.Header>
                  <h5>File Insurance Claim</h5>
                </Card.Header>
                <Card.Body>
                  {message && <Alert variant="success">{message}</Alert>}
                  {error && <Alert variant="danger">{error}</Alert>}
                  
                  <Form onSubmit={handleFileClaim}>
                    <Row>
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Policy ID</Form.Label>
                          <Form.Control
                            type="number"
                            name="policyId"
                            value={claimData.policyId}
                            onChange={handleClaimChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Claim Number</Form.Label>
                          <Form.Control
                            type="text"
                            name="claimNumber"
                            value={claimData.claimNumber}
                            onChange={handleClaimChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Claim Date</Form.Label>
                          <Form.Control
                            type="date"
                            name="claimDate"
                            value={claimData.claimDate}
                            onChange={handleClaimChange}
                            required
                          />
                        </Form.Group>
                      </Col>
                      
                      <Col md={6}>
                        <Form.Group className="mb-3">
                          <Form.Label>Incident Date</Form.Label>
                          <Form.Control
                            type="date"
                            name="incidentDate"
                            value={claimData.incidentDate}
                            onChange={handleClaimChange}
                            required
                          />
                        </Form.Group>
                        
                        <Form.Group className="mb-3">
                          <Form.Label>Claim Amount ($)</Form.Label>
                          <Form.Control
                            type="number"
                            step="0.01"
                            name="claimAmount"
                            value={claimData.claimAmount}
                            onChange={handleClaimChange}
                            required
                          />
                        </Form.Group>
                      </Col>
                    </Row>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Description</Form.Label>
                      <Form.Control
                        as="textarea"
                        rows={4}
                        name="description"
                        value={claimData.description}
                        onChange={handleClaimChange}
                        placeholder="Describe the incident and claim details..."
                        required
                      />
                    </Form.Group>
                    
                    <Button type="submit" variant="primary" disabled={loading} className="w-100">
                      {loading ? 'Filing Claim...' : 'File Claim'}
                    </Button>
                  </Form>
                </Card.Body>
              </Card>
            </Tab>
            
            <Tab eventKey="view" title="View Claims">
              <Card>
                <Card.Header>
                  <h5>View Claims by Policy</h5>
                </Card.Header>
                <Card.Body>
                  {error && <Alert variant="danger">{error}</Alert>}
                  
                  <Row className="mb-4">
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label>Policy ID</Form.Label>
                        <div className="d-flex">
                          <Form.Control
                            type="number"
                            value={policyId}
                            onChange={(e) => setPolicyId(e.target.value)}
                            placeholder="Enter Policy ID"
                          />
                          <Button 
                            variant="primary" 
                            className="ms-2"
                            onClick={handleGetClaims}
                            disabled={!policyId || loading}
                          >
                            Get Claims
                          </Button>
                        </div>
                      </Form.Group>
                    </Col>
                  </Row>
                  
                  {claims.length > 0 && (
                    <Table striped bordered hover responsive>
                      <thead>
                        <tr>
                          <th>Claim Number</th>
                          <th>Claim Date</th>
                          <th>Incident Date</th>
                          <th>Amount</th>
                          <th>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {claims.map((claim, index) => (
                          <tr key={index}>
                            <td>{claim.claimNumber}</td>
                            <td>{new Date(claim.claimDate).toLocaleDateString()}</td>
                            <td>{new Date(claim.incidentDate).toLocaleDateString()}</td>
                            <td>${claim.claimAmount}</td>
                            <td>{claim.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  )}
                  
                  {claims.length === 0 && policyId && !loading && (
                    <Alert variant="info">No claims found for this policy.</Alert>
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

export default Claims;
