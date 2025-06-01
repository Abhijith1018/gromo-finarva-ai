import React, { useState, useEffect, useContext } from 'react';
import { Container, Table, Card, Badge, Button, Alert } from 'react-bootstrap';
import NotificationService from '../../services/NotificationService';
import { AuthContext } from '../../utils/AuthContext';

const Notifications = () => {
    const [notifications, setNotifications] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const { user } = useContext(AuthContext);

    const fetchNotifications = async () => {
        if (!user) return;

        try {
            setLoading(true);
            const data = await NotificationService.getUserNotifications(user.id);
            setNotifications(data);
            setLoading(false);
        } catch (err) {
            setError('Failed to fetch notifications');
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchNotifications();
    }, [user]);

    const markAsRead = async (notificationId) => {
        try {
            await NotificationService.markNotificationAsRead(notificationId);
            setNotifications(notifications.map(n =>
                n.id === notificationId ? { ...n, read: true } : n
            ));
        } catch (err) {
            setError('Failed to mark notification as read');
        }
    };

    const markAllAsRead = async () => {
        try {
            setLoading(true);
            const unreadIds = notifications.filter(n => !n.read).map(n => n.id);

            for (const id of unreadIds) {
                await NotificationService.markNotificationAsRead(id);
            }

            setNotifications(notifications.map(n => ({ ...n, read: true })));
            setLoading(false);
        } catch (err) {
            setError('Failed to mark all notifications as read');
            setLoading(false);
        }
    };

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleString();
    };

    const unreadCount = notifications.filter(n => !n.read).length;

    return (
        <Container className="my-4">
            <Card>
                <Card.Header className="d-flex justify-content-between align-items-center">
                    <div>
                        <h2>Notifications</h2>
                        {unreadCount > 0 && (
                            <Badge bg="danger" pill>
                                {unreadCount} unread
                            </Badge>
                        )}
                    </div>
                    {unreadCount > 0 && (
                        <Button
                            variant="outline-primary"
                            size="sm"
                            onClick={markAllAsRead}
                            disabled={loading}
                        >
                            Mark All as Read
                        </Button>
                    )}
                </Card.Header>
                <Card.Body>
                    {error && <Alert variant="danger">{error}</Alert>}

                    {loading ? (
                        <div className="text-center my-4">Loading notifications...</div>
                    ) : notifications.length === 0 ? (
                        <Alert variant="info">You have no notifications</Alert>
                    ) : (
                        <Table responsive striped hover>
                            <thead>
                            <tr>
                                <th>Date & Time</th>
                                <th>Message</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                            </thead>
                            <tbody>
                            {notifications.map((notification) => (
                                <tr key={notification.id} className={notification.read ? "" : "table-active"}>
                                    <td>{formatDate(notification.timestamp)}</td>
                                    <td>{notification.message}</td>
                                    <td>
                                        <Badge bg={notification.read ? "secondary" : "danger"}>
                                            {notification.read ? "Read" : "Unread"}
                                        </Badge>
                                    </td>
                                    <td>
                                        {!notification.read && (
                                            <Button
                                                variant="outline-success"
                                                size="sm"
                                                onClick={() => markAsRead(notification.id)}
                                                disabled={loading}
                                            >
                                                Mark as Read
                                            </Button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                        </Table>
                    )}
                </Card.Body>
            </Card>
        </Container>
    );
};

export default Notifications;