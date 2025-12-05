import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useAuthStore } from './store/authStore';

// Components
import Login from './components/Auth/Login';
import Dashboard from './components/Dashboard/Dashboard';
import ContentReview from './components/ContentReview/ContentReview';
import AppealsManagement from './components/Appeals/AppealsManagement';
import UserAppeals from './components/Appeals/UserAppeals';
import Analytics from './components/Analytics/Analytics';
import Layout from './components/Layout/Layout';
import Stories from './components/Stories/Stories';
import StoryDetail from './components/Stories/StoryDetail';
import CommunityDashboard from './components/CommunityDashboard/CommunityDashboard';
import UserManagement from './components/Admin/UserManagement';
import UserSettings from './components/Settings/UserSettings';

// Theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#e33371',
      dark: '#9a0036',
    },
    success: {
      main: '#2e7d32',
      light: '#4caf50',
      50: '#e8f5e9',
    },
    warning: {
      main: '#ed6c02',
      light: '#ff9800',
      50: '#fff3e0',
    },
    error: {
      main: '#d32f2f',
      light: '#ef5350',
      50: '#ffebee',
    },
    info: {
      main: '#0288d1',
      light: '#03a9f4',
      50: '#e1f5fe',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
  },
});

// Protected Route Component
function ProtectedRoute({ children, requiredRoles = null }) {
  const { isAuthenticated, user } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check role-based access if requiredRoles specified
  if (requiredRoles && !requiredRoles.includes(user?.role)) {
    // Redirect to appropriate page based on role
    if (user?.role === 'user') {
      return <Navigate to="/portal" replace />;
    }
    return <Navigate to="/dashboard" replace />;
  }

  return <Layout>{children}</Layout>;
}

// User Route - for regular users (content submitters)
function UserRoute({ children }) {
  const { isAuthenticated, user } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Layout>{children}</Layout>;
}

// Role-based redirect after login
function RoleBasedRedirect() {
  const { isAuthenticated, user } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Redirect based on role - everyone goes to community dashboard first
  if (user?.role === 'user') {
    return <Navigate to="/community" replace />;
  }

  // Moderators and above can access moderator dashboard or community
  return <Navigate to="/community" replace />;
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />

          {/* Community Dashboard - Main landing page for all authenticated users */}
          <Route
            path="/community"
            element={
              <UserRoute>
                <CommunityDashboard />
              </UserRoute>
            }
          />

          {/* Stories - Dedicated stories feed view */}
          <Route
            path="/stories"
            element={
              <UserRoute>
                <Stories />
              </UserRoute>
            }
          />

          {/* Story Detail - view individual story with comments */}
          <Route
            path="/stories/:storyId"
            element={
              <UserRoute>
                <StoryDetail />
              </UserRoute>
            }
          />

          {/* My Appeals - for all users to submit and track appeals */}
          <Route
            path="/my-appeals"
            element={
              <UserRoute>
                <UserAppeals />
              </UserRoute>
            }
          />

          {/* Moderator Dashboard */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute requiredRoles={['moderator', 'senior_moderator', 'content_analyst', 'policy_specialist', 'admin']}>
                <Dashboard />
              </ProtectedRoute>
            }
          />

          {/* Content Review - for moderators */}
          <Route
            path="/review/:contentId"
            element={
              <ProtectedRoute requiredRoles={['moderator', 'senior_moderator', 'content_analyst', 'policy_specialist', 'admin']}>
                <ContentReview />
              </ProtectedRoute>
            }
          />

          {/* Appeals Management - for policy specialists and admins */}
          <Route
            path="/appeals"
            element={
              <ProtectedRoute requiredRoles={['policy_specialist', 'admin']}>
                <AppealsManagement />
              </ProtectedRoute>
            }
          />

          {/* Analytics - for analysts and above */}
          <Route
            path="/analytics"
            element={
              <ProtectedRoute requiredRoles={['content_analyst', 'senior_moderator', 'policy_specialist', 'admin']}>
                <Analytics />
              </ProtectedRoute>
            }
          />

          {/* User Management - for admins only */}
          <Route
            path="/user-management"
            element={
              <ProtectedRoute requiredRoles={['admin']}>
                <UserManagement />
              </ProtectedRoute>
            }
          />

          {/* User Settings - for all authenticated users */}
          <Route
            path="/settings"
            element={
              <UserRoute>
                <UserSettings />
              </UserRoute>
            }
          />

          {/* Default redirect based on role */}
          <Route path="/" element={<RoleBasedRedirect />} />

          {/* Catch-all redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
