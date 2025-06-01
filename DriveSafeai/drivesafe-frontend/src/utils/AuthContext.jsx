import React, { createContext, useState, useEffect } from 'react';
import UserService from '../services/UserService.js';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Check if there's a token in localStorage
        const token = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');

        const verifyUser = async () => {
            if (token && userId) {
                try {
                    // Fetch user details with the stored userId
                    const userData = await UserService.getUserById(userId);
                    setUser(userData);
                } catch (err) {
                    console.error('Error verifying user:', err);
                    localStorage.removeItem('token');
                    localStorage.removeItem('userId');
                }
            }
            setLoading(false);
        };

        verifyUser();
    }, []);

    const login = async (credentials) => {
        try {
            setLoading(true);
            const token = await UserService.login(credentials);

            // Save token to localStorage
            localStorage.setItem('token', token);

            // Extract userId from token (assuming JWT with userId in payload)
            // In a real app, you might decode the JWT to get the userId
            // For now, we'll assume the login response contains the userId
            const userId = getUserIdFromToken(token);
            localStorage.setItem('userId', userId);

            // Fetch user details
            const userData = await UserService.getUserById(userId);
            setUser(userData);
            setError(null);
            setLoading(false);
            return true;
        } catch (err) {
            setError(err.message || 'Login failed');
            setLoading(false);
            return false;
        }
    };

    const register = async (userData) => {
        try {
            setLoading(true);
            const registeredUser = await UserService.register(userData);
            setLoading(false);
            return registeredUser;
        } catch (err) {
            setError(err.message || 'Registration failed');
            setLoading(false);
            throw err;
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('userId');
        setUser(null);
    };

    // Helper function to extract userId from token
    // In a real app, you would use a proper JWT decoder library
    const getUserIdFromToken = (token) => {
        // This is a simplified implementation.
        // Normally, you would decode the JWT and extract the userId from the payload
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));

            return JSON.parse(jsonPayload).userId || JSON.parse(jsonPayload).sub;
        } catch (e) {
            console.error("Error extracting userId from token:", e);
            return null;
        }
    };

    return (
        <AuthContext.Provider value={{
            user,
            loading,
            error,
            login,
            register,
            logout,
            isAuthenticated: !!user
        }}>
            {children}
        </AuthContext.Provider>
    );
};