import axios from 'axios';

const API_URL = 'http://localhost:8080/api';

// Helper method for auth headers
const getAuthHeader = () => {
    const token = localStorage.getItem('token');
    return {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    };
};

const TripService = {
    live: async (obdData) => {
        try {
            const response = await axios.post(`${API_URL}/live`, obdData, getAuthHeader());
            return response.data;
        } catch (error) {
            throw new Error(error.response?.data?.message || 'Failed to send live trip data');
        }
    },

    endSession: async (sessionId) => {
        try {
            const response = await axios.post(`${API_URL}/end-session/${sessionId}`, {}, getAuthHeader());
            return response.data;
        } catch (error) {
            throw new Error(error.response?.data?.message || 'Failed to end session');
        }
    },

    tripHistory: async (userId) => {
        try {
            const response = await axios.get(`${API_URL}/trip-summary/${userId}`, getAuthHeader());
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.message || 'Failed to fetch Trip history');
        }
    },
};

export default TripService;