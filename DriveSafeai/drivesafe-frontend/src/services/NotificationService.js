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

const NotificationService = {
    getUserNotifications: async (userId) => {
        try {
            const response = await axios.get(`${API_URL}/notifications/${userId}`, getAuthHeader());
            return response.data;
        } catch (error) {
            throw new Error(error.response?.data?.message || 'Failed to get notifications');
        }
    },

    markNotificationAsRead: async (notificationId) => {
        try {
            const response = await axios.post(`${API_URL}/notifications/${notificationId}/mark-read`,{}, getAuthHeader());
            return response.data;
        } catch (error) {
            throw new Error(error.response?.data?.message || 'Failed to mark notification as read');
        }
    }
};

export default NotificationService;