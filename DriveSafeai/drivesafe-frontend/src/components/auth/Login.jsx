import {useContext, useState} from 'react';
import { useNavigate, Link } from 'react-router-dom';
import UserService from '../../services/UserService';
import { Form, Button, Container, Row, Col, Card, Alert } from 'react-bootstrap';
import {AuthContext} from "../../utils/AuthContext.jsx";

const Login = () => {
  const navigate = useNavigate();
  const { login } = useContext(AuthContext);
  const [credentials, setCredentials] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCredentials(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await login(credentials);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to login. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const containerStyle = {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  };

  const cardStyle = {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(20px)',
    border: 'none',
    borderRadius: '24px',
    boxShadow: '0 25px 45px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.1)',
    transform: 'translateY(0)',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    overflow: 'hidden',
    position: 'relative'
  };

  const cardHoverStyle = {
    ...cardStyle,
    transform: 'translateY(-5px)',
    boxShadow: '0 35px 55px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.15)'
  };

  const headerStyle = {
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    fontSize: '2.5rem',
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: '2rem',
    letterSpacing: '-0.5px'
  };

  const formGroupStyle = {
    marginBottom: '1.5rem',
    position: 'relative'
  };

  const labelStyle = {
    fontWeight: '600',
    color: '#374151',
    marginBottom: '0.5rem',
    fontSize: '0.95rem',
    letterSpacing: '0.3px'
  };

  const inputBaseStyle = {
    border: '2px solid #e5e7eb',
    borderRadius: '12px',
    padding: '12px 16px',
    fontSize: '1rem',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    background: 'rgba(255, 255, 255, 0.8)',
    backdropFilter: 'blur(10px)',
    outline: 'none',
    width: '100%'
  };

  const getInputStyle = (fieldName) => ({
    ...inputBaseStyle,
    borderColor: focusedField === fieldName ? '#667eea' : (credentials[fieldName] ? '#10b981' : '#e5e7eb'),
    boxShadow: focusedField === fieldName
        ? '0 0 0 3px rgba(102, 126, 234, 0.1), 0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        : credentials[fieldName]
            ? '0 0 0 3px rgba(16, 185, 129, 0.1)'
            : 'none',
    transform: focusedField === fieldName ? 'translateY(-1px)' : 'translateY(0)'
  });

  const buttonStyle = {
    background: loading
        ? 'linear-gradient(135deg, #9ca3af, #6b7280)'
        : 'linear-gradient(135deg, #667eea, #764ba2)',
    border: 'none',
    borderRadius: '12px',
    padding: '14px 28px',
    fontSize: '1.1rem',
    fontWeight: '600',
    color: 'white',
    cursor: loading ? 'not-allowed' : 'pointer',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    transform: 'translateY(0)',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    width: '100%',
    marginTop: '1rem',
    position: 'relative',
    overflow: 'hidden'
  };

  const buttonHoverStyle = {
    ...buttonStyle,
    transform: loading ? 'translateY(0)' : 'translateY(-2px)',
    boxShadow: loading
        ? '0 4px 15px rgba(102, 126, 234, 0.4)'
        : '0 8px 25px rgba(102, 126, 234, 0.6)',
    background: loading
        ? 'linear-gradient(135deg, #9ca3af, #6b7280)'
        : 'linear-gradient(135deg, #5a67d8, #6b46c1)'
  };

  const linkStyle = {
    color: '#667eea',
    textDecoration: 'none',
    fontWeight: '600',
    transition: 'all 0.3s ease',
    borderBottom: '2px solid transparent'
  };

  const linkHoverStyle = {
    ...linkStyle,
    color: '#5a67d8',
    borderBottomColor: '#5a67d8'
  };

  const alertStyle = {
    background: 'linear-gradient(135deg, #fee2e2, #fecaca)',
    border: '1px solid #f87171',
    borderRadius: '12px',
    color: '#dc2626',
    padding: '12px 16px',
    marginBottom: '1.5rem',
    animation: 'slideIn 0.3s ease-out'
  };

  const [isHovered, setIsHovered] = useState(false);
  const [buttonHovered, setButtonHovered] = useState(false);
  const [linkHovered, setLinkHovered] = useState(false);

  return (
      <div style={containerStyle}>
        <style>
          {`
          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(-10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.8;
            }
          }
          
          .loading-animation {
            animation: pulse 1.5s ease-in-out infinite;
          }
        `}
        </style>
        <Container>
          <Row className="justify-content-center">
            <Col md={6} lg={5}>
              <Card
                  style={isHovered ? cardHoverStyle : cardStyle}
                  onMouseEnter={() => setIsHovered(true)}
                  onMouseLeave={() => setIsHovered(false)}
              >
                <Card.Body style={{ padding: '3rem 2.5rem' }}>
                  <h2 style={headerStyle}>DriveSafe Login</h2>

                  {error && (
                      <Alert style={alertStyle}>
                        <strong>⚠️ </strong>{error}
                      </Alert>
                  )}

                  <Form onSubmit={handleSubmit}>
                    <div style={formGroupStyle}>
                      <Form.Label style={labelStyle}>Email Address</Form.Label>
                      <Form.Control
                          type="email"
                          name="email"
                          value={credentials.email}
                          onChange={handleChange}
                          onFocus={() => setFocusedField('email')}
                          onBlur={() => setFocusedField('')}
                          required
                          placeholder="Enter your email address"
                          style={getInputStyle('email')}
                      />
                    </div>

                    <div style={formGroupStyle}>
                      <Form.Label style={labelStyle}>Password</Form.Label>
                      <Form.Control
                          type="password"
                          name="password"
                          value={credentials.password}
                          onChange={handleChange}
                          onFocus={() => setFocusedField('password')}
                          onBlur={() => setFocusedField('')}
                          required
                          placeholder="Enter your password"
                          style={getInputStyle('password')}
                      />
                    </div>

                    <Button
                        type="submit"
                        disabled={loading}
                        style={buttonHovered ? buttonHoverStyle : buttonStyle}
                        onMouseEnter={() => setButtonHovered(true)}
                        onMouseLeave={() => setButtonHovered(false)}
                        className={loading ? 'loading-animation' : ''}
                    >
                      {loading ? (
                          <>
                            <span style={{ marginRight: '8px' }}>⏳</span>
                            Signing you in...
                          </>
                      ) : (
                          <>
                            <span style={{ marginRight: '8px' }}>🚗</span>
                            Sign In to DriveSafe
                          </>
                      )}
                    </Button>
                  </Form>

                  <div style={{ textAlign: 'center', marginTop: '2rem', color: '#6b7280' }}>
                    <p style={{ margin: 0, fontSize: '0.95rem' }}>
                      Don't have an account?{' '}
                      <Link
                          to="/register"
                          style={linkHovered ? linkHoverStyle : linkStyle}
                          onMouseEnter={() => setLinkHovered(true)}
                          onMouseLeave={() => setLinkHovered(false)}
                      >
                        Create one here →
                      </Link>
                    </p>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Container>
      </div>
  );
};

export default Login;