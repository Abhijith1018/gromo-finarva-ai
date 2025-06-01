import React from 'react';
import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './utils/AuthContext';
import PrivateRoute from './utils/PrivateRoute';

// Common Components
import Header from './Components/Common/Header';
import Footer from './components/common/Footer';
import Notifications from './components/common/Notifications';

// Auth Components
import Login from './components/auth/Login';
import Register from './components/auth/Register';

// Dashboard Components
import Dashboard from './components/dashboard/Dashboard';
import OBDDashboard from "./pages/OBDSimulator.jsx";
import TripDashboard from "./components/dashboard/TripDashboard.jsx";
import RewardsPage from "./components/dashboard/RewardPointRedemption.jsx";
import Home from "./pages/Home.jsx";
import Insurance from "./components/dashboard/Insurance.jsx";


// Error Pages
const UnauthorizedPage = () => (
    <div className="container text-center py-5">
        <h1>Unauthorized Access</h1>
        <p>You don't have permission to access this page.</p>
    </div>
);

const NotFoundPage = () => (
    <div className="container text-center py-5">
        <h1>404 - Page Not Found</h1>
        <p>The page you are looking for does not exist.</p>
    </div>
);

// Landing Page
const HomePage = () => (
    <div className="container">
        <div className="row justify-content-center align-items-center py-5">
            <div className="col-md-6">
                <h1 className="display-4 mb-4">Welcome to DriveSafeAI</h1>
                <p className="lead">
                    AI-powered platform for driving safety, risk assessment, and insurance management.
                </p>
                <div className="mt-4">
                    <a href="/login" className="btn btn-primary me-2">Login</a>
                    <a href="/register" className="btn btn-outline-secondary">Register</a>
                </div>
            </div>
            <div className="col-md-6">
                <img src="./assets/drivesafe_logo.jpeg" alt="Driving safety illustration" className="img-fluid rounded shadow" />
            </div>
        </div>
    </div>
);

const App = () => {
    return (
        <AuthProvider>
            <Router>
                <div className="d-flex flex-column min-vh-100">
                    <Header />
                    <div className="flex-grow-1">
                        <Routes>
                            {/* Public Routes */}
                            <Route path="/" element={<Home />} />
                            <Route path="/login" element={<Login />} />
                            <Route path="/register" element={<Register />} />

                            {/* Protected Routes for All Users */}
                            <Route element={<PrivateRoute />}>
                                <Route path="/dashboard" element={<Dashboard />} />
                                <Route path="/notifications" element={<Notifications />} />
                                <Route path="/trip-monitor" element={<OBDDashboard />} />
                                <Route path="/trip-history" element={<TripDashboard />} />
                                <Route path="/rewards-page" element={<RewardsPage />} />
                                <Route path="/insurance" element={<Insurance />} />
                            </Route>


                            {/* Error Routes */}
                            <Route path="/unauthorized" element={<UnauthorizedPage />} />
                            <Route path="*" element={<NotFoundPage />} />
                        </Routes>
                    </div>
                    <Footer />
                </div>
            </Router>
        </AuthProvider>
    );
};

export default App;