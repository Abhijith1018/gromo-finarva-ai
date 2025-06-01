import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080/api';

const apiService = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor to add JWT token
apiService.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('authToken');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor to handle token expiration
apiService.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('jwtToken');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// Auth API calls
export const authAPI = {
    login: (credentials) => apiService.post('/login', credentials),
    register: (userData) => apiService.post('/register', userData),
};

// Trip API calls
export const tripAPI = {
    submitTrip: (tripData) => apiService.post('/trip', tripData),
};

// User API calls
export const userAPI = {
    getDriscScore: (userId) => apiService.get(`/drisc-score/${userId}`),
    getNotifications: (userId) => apiService.get(`/notifications/${userId}`),
};

// Insurance API calls
export const insuranceAPI = {
    createPolicy: (policyData) => apiService.post('/insurance/policy', policyData),
    calculatePremium: (userId) => apiService.get(`/insurance/premium/${userId}`),
    fileClaim: (claimData) => apiService.post('/insurance/claim', claimData),
    getClaimsByPolicy: (policyId) => apiService.get(`/insurance/claim/${policyId}`),
};

export default apiService;
