import React, { useContext } from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { AuthContext } from './AuthContext';
import { Spinner } from 'react-bootstrap';

const PrivateRoute = ({ requiredRole }) => {
    const { user, loading, isAuthenticated } = useContext(AuthContext);
    const location = useLocation();

    if (loading) {
        return (
            <div className="d-flex justify-content-center align-items-center" style={{ height: "100vh" }}>
                <Spinner animation="border" role="status">
                    <span className="visually-hidden">Loading...</span>
                </Spinner>
            </div>
        );
    }

    // Check if user is authenticated
    if (!isAuthenticated) {
        // Redirect to login page with the return url
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // If no specific role is required, or user has the required role
    if (!requiredRole || user.role === requiredRole) {
        return <Outlet />;
    }

    // If user doesn't have the required role
    return <Navigate to="/unauthorized" replace />;
};

export default PrivateRoute;