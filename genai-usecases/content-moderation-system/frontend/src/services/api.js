import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors (unauthorized)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Authentication API
export const authAPI = {
  login: async (username, password) => {
    const response = await api.post('/api/auth/login', { username, password });
    const { token, user } = response.data;
    localStorage.setItem('token', token);
    localStorage.setItem('user', JSON.stringify(user));
    return response.data;
  },

  logout: async () => {
    try {
      await api.post('/api/auth/logout');
    } finally {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
  },

  getCurrentUser: async () => {
    const response = await api.get('/api/auth/current-user');
    return response.data;
  },

  register: async (userData) => {
    const response = await api.post('/api/auth/register', userData);
    return response.data;
  },

  updatePassword: async (currentPassword, newPassword) => {
    const response = await api.put('/api/auth/password', {
      current_password: currentPassword,
      new_password: newPassword
    });
    return response.data;
  },

  getAllUsers: async () => {
    const response = await api.get('/api/auth/users');
    return response.data;
  },

  updateUser: async (userId, userData) => {
    const response = await api.put(`/api/auth/users/${userId}`, userData);
    return response.data;
  },

  deleteUser: async (userId) => {
    const response = await api.delete(`/api/auth/users/${userId}`);
    return response.data;
  },
};

// Content Moderation API
export const contentAPI = {
  submitContent: async (contentData) => {
    const response = await api.post('/api/content/submit', contentData);
    return response.data;
  },

  getPendingContent: async (limit = 20) => {
    const response = await api.get('/api/content/pending', { params: { limit } });
    return response.data;
  },

  getContentById: async (contentId) => {
    const response = await api.get(`/api/content/${contentId}`);
    return response.data;
  },

  getAllContent: async (status = null, limit = 100) => {
    const params = { limit };
    if (status) params.status = status;
    const response = await api.get('/api/content/all', { params });
    return response.data;
  },

  moderateContent: async (contentId) => {
    const response = await api.post(`/api/content/${contentId}/moderate`);
    return response.data;
  },

  getContentStatus: async (contentId) => {
    const response = await api.get(`/api/content/${contentId}/status`);
    return response.data;
  },
};

// Appeals API
export const appealsAPI = {
  submitAppeal: async (appealData) => {
    const response = await api.post('/api/appeals/submit', appealData);
    return response.data;
  },

  createAppeal: async (appealData) => {
    const response = await api.post('/api/appeals/submit', appealData);
    return response.data;
  },

  getPendingAppeals: async (limit = 20) => {
    const response = await api.get('/api/appeals/pending', { params: { limit } });
    return response.data;
  },

  getAllAppeals: async (status = null, limit = 100) => {
    const params = { limit };
    if (status) params.status = status;
    const response = await api.get('/api/appeals/all', { params });
    return response.data;
  },

  getUserAppeals: async (userId) => {
    const response = await api.get(`/api/appeals/user/${userId}`);
    return response.data;
  },

  reviewAppeal: async (appealId, reviewData) => {
    const response = await api.post(`/api/appeals/${appealId}/review`, reviewData);
    return response.data;
  },

  getAppealById: async (appealId) => {
    const response = await api.get(`/api/appeals/${appealId}`);
    return response.data;
  },

  getAppealsByContent: async (contentId) => {
    const response = await api.get(`/api/appeals/content/${contentId}`);
    return response.data;
  },
};

// Analytics API
export const analyticsAPI = {
  getSystemMetrics: async () => {
    const response = await api.get('/api/analytics/metrics');
    return response.data;
  },

  getAgentPerformance: async (agentName = null) => {
    const params = agentName ? { agent_name: agentName } : {};
    const response = await api.get('/api/analytics/agent-performance', { params });
    return response.data;
  },

  getLearningMetrics: async (agentName = null, days = 30) => {
    const params = { days };
    if (agentName) params.agent_name = agentName;
    const response = await api.get('/api/analytics/learning', { params });
    return response.data;
  },

  getDecisionHistory: async (agentName = null, limit = 100) => {
    const params = { limit };
    if (agentName) params.agent_name = agentName;
    const response = await api.get('/api/analytics/decisions', { params });
    return response.data;
  },

  getAppealTrends: async (days = 30) => {
    const response = await api.get('/api/analytics/appeal-trends', { params: { days } });
    return response.data;
  },
};

// Users API
export const usersAPI = {
  getUserReputation: async (userId) => {
    const response = await api.get(`/api/users/${userId}/reputation`);
    return response.data;
  },

  getUserHistory: async (userId, limit = 50) => {
    const response = await api.get(`/api/users/${userId}/history`, { params: { limit } });
    return response.data;
  },

  updateUserStatus: async (userId, status) => {
    const response = await api.post(`/api/users/${userId}/status`, { status });
    return response.data;
  },
};

// Stories API
export const storiesAPI = {
  submitStory: async (storyData) => {
    const response = await api.post('/api/stories/submit', storyData);
    return response.data;
  },

  getStories: async (visibleOnly = true, limit = 50) => {
    const response = await api.get('/api/stories', { params: { visible_only: visibleOnly, limit } });
    return response.data;
  },

  getStoryById: async (storyId, includeComments = true) => {
    const response = await api.get(`/api/stories/${storyId}`, { params: { include_comments: includeComments } });
    return response.data;
  },

  getUserStories: async (userId, limit = 50) => {
    const response = await api.get(`/api/stories/user/${userId}`, { params: { limit } });
    return response.data;
  },

  getPendingStories: async (limit = 50) => {
    const response = await api.get('/api/stories/pending', { params: { limit } });
    return response.data;
  },

  submitComment: async (storyId, commentData) => {
    const response = await api.post(`/api/stories/${storyId}/comments`, commentData);
    return response.data;
  },

  getComments: async (storyId, visibleOnly = true) => {
    const response = await api.get(`/api/stories/${storyId}/comments`, { params: { visible_only: visibleOnly } });
    return response.data;
  },

  getPendingComments: async (limit = 50) => {
    const response = await api.get('/api/stories/comments/pending', { params: { limit } });
    return response.data;
  },
};

export default api;
