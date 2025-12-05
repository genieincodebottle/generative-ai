import { create } from 'zustand';

export const useAuthStore = create((set, get) => ({
  user: JSON.parse(localStorage.getItem('user')) || null,
  token: localStorage.getItem('token') || null,
  isAuthenticated: !!localStorage.getItem('token'),

  login: (user, token) => {
    localStorage.setItem('user', JSON.stringify(user));
    localStorage.setItem('token', token);
    set({ user, token, isAuthenticated: true });
  },

  logout: () => {
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    set({ user: null, token: null, isAuthenticated: false });
  },

  updateUser: (user) => {
    localStorage.setItem('user', JSON.stringify(user));
    set({ user });
  },

  // Helper functions - Role checks
  isUser: () => {
    const { user } = get();
    return user?.role === 'user';
  },

  isModerator: () => {
    const { user } = get();
    const modRoles = ['moderator', 'senior_moderator', 'content_analyst', 'policy_specialist', 'admin'];
    return modRoles.includes(user?.role);
  },

  isSeniorModerator: () => {
    const { user } = get();
    const seniorRoles = ['senior_moderator', 'policy_specialist', 'admin'];
    return seniorRoles.includes(user?.role);
  },

  isAdmin: () => {
    const { user } = get();
    return user?.role === 'admin';
  },

  // Permission checks
  canSubmitContent: () => {
    // All authenticated users can submit content
    const { user } = get();
    return !!user;
  },

  canReviewContent: () => {
    const { user } = get();
    const reviewRoles = ['moderator', 'senior_moderator', 'policy_specialist', 'admin'];
    return reviewRoles.includes(user?.role);
  },

  canReviewAppeals: () => {
    const { user } = get();
    const appealRoles = ['policy_specialist', 'admin'];
    return appealRoles.includes(user?.role);
  },

  canReviewHITL: () => {
    const { user } = get();
    const hitlRoles = ['senior_moderator', 'policy_specialist', 'admin'];
    return hitlRoles.includes(user?.role);
  },

  canViewAnalytics: () => {
    const { user } = get();
    const analyticsRoles = ['content_analyst', 'senior_moderator', 'policy_specialist', 'admin'];
    return analyticsRoles.includes(user?.role);
  },

  canManageUsers: () => {
    const { user } = get();
    return user?.role === 'admin';
  },
}));
