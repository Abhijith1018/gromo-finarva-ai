import React, { useContext } from 'react';
import { Navbar, Nav, Container, Button, NavDropdown, Badge, Image } from 'react-bootstrap';
import { Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../../utils/AuthContext';
import NotificationService from '../../services/NotificationService';
import UserService from '../../services/UserService';

const Header = () => {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();
  const [notifications, setNotifications] = React.useState([]);
  const [unreadCount, setUnreadCount] = React.useState(0);
  const [rewardPoints, setRewardPoints] = React.useState(0);

  React.useEffect(() => {
    const fetchNotifications = async () => {
      if (user) {
        try {
          const id = localStorage.getItem("userId");
          const data = await NotificationService.getUserNotifications(id);
          setNotifications(data);
          setUnreadCount(data.filter(n => !n.read).length);
        } catch (error) {
          console.error('Failed to fetch notifications:', error);
        }
      }
    };

    const fetchRewardPoints = async () => {
      if (user) {
        try {
          const userId = localStorage.getItem("userId");
          const points = await UserService.getRewardPoints(userId);
          setRewardPoints(points);
        } catch (error) {
          console.error('Failed to fetch reward points:', error);
        }
      }
    };

    fetchNotifications();
    fetchRewardPoints();

    // Set up polling for notifications and reward points every minute
    const interval = setInterval(() => {
      fetchNotifications();
      fetchRewardPoints();
    }, 60000);

    return () => clearInterval(interval);
  }, [user]);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const markAsRead = async (notificationId) => {
    try {
      await NotificationService.markNotificationAsRead(notificationId);
      setNotifications(notifications.map(n =>
          n.id === notificationId ? { ...n, read: true } : n
      ));
      setUnreadCount(prev => prev - 1);
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
    }
  };

  return (
      <Navbar
          expand="lg"
          sticky="top"
          style={{
            background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
          }}
          variant="dark"
      >
        <Container>
          <Navbar.Brand
              as={Link}
              to="/"
              style={{
                fontSize: '1.8rem',
                fontWeight: 'bold',
                background: 'linear-gradient(45deg, #00d4ff, #007bff)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: '0 0 20px rgba(0, 123, 255, 0.3)',
                letterSpacing: '1px'
              }}
          >
            🚗 DriveSafeAI
          </Navbar.Brand>

          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link
                  as={Link}
                  to="/"
                  style={{
                    color: '#ffffff',
                    fontWeight: '500',
                    padding: '0.5rem 1rem',
                    borderRadius: '8px',
                    transition: 'all 0.3s ease',
                    position: 'relative'
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                    e.target.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'transparent';
                    e.target.style.transform = 'translateY(0)';
                  }}
              >
                🏠 Home
              </Nav.Link>
              {user && (
                  <>
                    <Nav.Link
                        as={Link}
                        to="/dashboard"
                        style={{
                          color: '#ffffff',
                          fontWeight: '500',
                          padding: '0.5rem 1rem',
                          borderRadius: '8px',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                          e.target.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'transparent';
                          e.target.style.transform = 'translateY(0)';
                        }}
                    >
                      📊 Dashboard
                    </Nav.Link>
                    <Nav.Link
                        as={Link}
                        to="/trip-history"
                        style={{
                          color: '#ffffff',
                          fontWeight: '500',
                          padding: '0.5rem 1rem',
                          borderRadius: '8px',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                          e.target.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'transparent';
                          e.target.style.transform = 'translateY(0)';
                        }}
                    >
                      🛣️ Trip History
                    </Nav.Link>
                    <Nav.Link
                        as={Link}
                        to="/insurance"
                        style={{
                          color: '#ffffff',
                          fontWeight: '500',
                          padding: '0.5rem 1rem',
                          borderRadius: '8px',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                          e.target.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'transparent';
                          e.target.style.transform = 'translateY(0)';
                        }}
                    >
                      🛡️ Insurance
                    </Nav.Link>
                  </>
              )}
            </Nav>

            <Nav className="align-items-center">
              {user ? (
                  <>
                    <NavDropdown
                        title={
                          <span style={{ color: '#ffffff' }}>
                            🔔 Notifications
                            {unreadCount > 0 && (
                                <Badge
                                    bg="danger"
                                    pill
                                    className="ms-1"
                                    style={{
                                      animation: 'pulse 2s infinite',
                                      boxShadow: '0 0 10px rgba(220, 53, 69, 0.6)'
                                    }}
                                >
                                  {unreadCount}
                                </Badge>
                            )}
                          </span>
                        }
                        id="notifications-dropdown"
                        className="me-3"
                        style={{
                          background: 'rgba(255, 255, 255, 0.1)',
                          borderRadius: '8px',
                          backdropFilter: 'blur(10px)'
                        }}
                    >
                      {notifications.length === 0 ? (
                          <NavDropdown.Item disabled>No notifications</NavDropdown.Item>
                      ) : (
                          <>
                            {notifications.slice(0, 5).map(notification => (
                                <NavDropdown.Item
                                    key={notification.id}
                                    onClick={() => markAsRead(notification.id)}
                                    className={notification.read ? "" : "fw-bold"}
                                    style={{
                                      borderRadius: '6px',
                                      margin: '2px',
                                      background: notification.read ? 'transparent' : 'rgba(0, 123, 255, 0.1)'
                                    }}
                                >
                                  {notification.message}
                                  <div className="text-muted small">
                                    {new Date(notification.timestamp).toLocaleString()}
                                  </div>
                                </NavDropdown.Item>
                            ))}
                            <NavDropdown.Divider />
                            <NavDropdown.Item as={Link} to="/notifications">
                              View All Notifications
                            </NavDropdown.Item>
                          </>
                      )}
                    </NavDropdown>

                      {/* Reward Points Section */}
                      <div
                          className="me-3 d-flex align-items-center"
                          style={{
                              background: 'linear-gradient(135deg, #ffd700 0%, #ffb347 100%)',
                              color: '#1a1a2e',
                              padding: '8px 16px',
                              borderRadius: '25px',
                              fontWeight: 'bold',
                              fontSize: '0.95rem',
                              boxShadow: '0 4px 15px rgba(255, 215, 0, 0.3)',
                              border: '2px solid rgba(255, 215, 0, 0.5)',
                              transition: 'all 0.3s ease',
                              cursor: 'pointer'
                          }}
                          onMouseEnter={(e) => {
                              e.currentTarget.style.transform = 'scale(1.05)';
                              e.currentTarget.style.boxShadow = '0 6px 20px rgba(255, 215, 0, 0.4)';
                          }}
                          onMouseLeave={(e) => {
                              e.currentTarget.style.transform = 'scale(1)';
                              e.currentTarget.style.boxShadow = '0 4px 15px rgba(255, 215, 0, 0.3)';
                          }}
                          onClick={() => window.location.href = '/rewards-page'}
                          title="Your Reward Points - Click to view rewards"
                      >
                          <span
                              style={{
                                  fontSize: '1.2rem',
                                  marginRight: '6px',
                                  animation: 'spin 3s linear infinite'
                              }}
                          >
                            🪙
                          </span>
                          {rewardPoints.toLocaleString()}
                      </div>

                    <Navbar.Text
                        className="me-3"
                        style={{
                          color: '#ffffff',
                          fontWeight: '500',
                          padding: '8px 12px',
                          background: 'rgba(255, 255, 255, 0.1)',
                          borderRadius: '20px',
                          backdropFilter: 'blur(10px)',
                          border: '1px solid rgba(255, 255, 255, 0.2)'
                        }}
                    >
                      👋 Welcome, {user?.fullName}
                    </Navbar.Text>

                    <Button
                        variant="outline-light"
                        onClick={handleLogout}
                        style={{
                          borderRadius: '25px',
                          padding: '8px 20px',
                          fontWeight: '600',
                          border: '2px solid rgba(255, 255, 255, 0.3)',
                          background: 'rgba(255, 255, 255, 0.1)',
                          backdropFilter: 'blur(10px)',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(255, 255, 255, 0.2)';
                          e.target.style.transform = 'translateY(-2px)';
                          e.target.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.2)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                          e.target.style.transform = 'translateY(0)';
                          e.target.style.boxShadow = 'none';
                        }}
                    >
                      Logout
                    </Button>
                  </>
              ) : (
                  <>
                    <Nav.Link
                        as={Link}
                        to="/login"
                        style={{
                          color: '#ffffff',
                          fontWeight: '500',
                          padding: '8px 16px',
                          borderRadius: '20px',
                          marginRight: '10px',
                          background: 'rgba(0, 123, 255, 0.2)',
                          border: '1px solid rgba(0, 123, 255, 0.5)',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(0, 123, 255, 0.3)';
                          e.target.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'rgba(0, 123, 255, 0.2)';
                          e.target.style.transform = 'translateY(0)';
                        }}
                    >
                      Login
                    </Nav.Link>
                    <Nav.Link
                        as={Link}
                        to="/register"
                        style={{
                          color: '#ffffff',
                          fontWeight: '500',
                          padding: '8px 16px',
                          borderRadius: '20px',
                          background: 'rgba(40, 167, 69, 0.2)',
                          border: '1px solid rgba(40, 167, 69, 0.5)',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(40, 167, 69, 0.3)';
                          e.target.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'rgba(40, 167, 69, 0.2)';
                          e.target.style.transform = 'translateY(0)';
                        }}
                    >
                      Register
                    </Nav.Link>
                  </>
              )}
            </Nav>
          </Navbar.Collapse>
        </Container>

        <style>
          {`
            @keyframes pulse {
              0% { transform: scale(1); }
              50% { transform: scale(1.1); }
              100% { transform: scale(1); }
            }
            
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }

            .navbar-nav .nav-link:hover {
              text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
            }

            .dropdown-menu {
              background: rgba(26, 26, 46, 0.95) !important;
              backdrop-filter: blur(10px);
              border: 1px solid rgba(255, 255, 255, 0.1);
              border-radius: 12px;
              box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }

            .dropdown-item {
              color: #ffffff !important;
              transition: all 0.3s ease;
            }

            .dropdown-item:hover {
              background: rgba(0, 123, 255, 0.2) !important;
              color: #ffffff !important;
            }
          `}
        </style>
      </Navbar>
  );
};

export default Header;