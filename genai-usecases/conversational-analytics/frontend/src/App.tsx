import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material';
import { Container, CircularProgress, Box, CssBaseline } from '@mui/material';
import Register from './components/Register';
import Login from './components/Login';
import Home from './components/Home';
import Header from './components/Header';
import Footer from './components/Footer';
import PrivateRoute from './components/PrivateRoute';
import FeedbackForm from './components/FeedbackForm';
import AnalyticsDisplay from './components/AnalyticsDisplay';
import FeedbackDisplay from './components/FeedbackDisplay';
import { Analytics, FeedbackData, UserRole, API_BASE_URL } from './types';

const theme = createTheme({
  palette: {
    primary: { main: '#05a6eb' },
    secondary: { main: '#05a6eb' },
  },
});

const App: React.FC = () => {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [feedback, setFeedback] = useState<FeedbackData | null>(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [userRole, setUserRole] = useState<UserRole | null>(null);
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    const handleStorageChange = () => setToken(localStorage.getItem('token'));
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  useEffect(() => {
    if (token) {
      fetchUserData();
      fetchData();
    } else {
      setLoading(false);
      setUserRole(null);
      setUsername(null);
    }
  }, [token]);

  const getAuthHeaders = () => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  });

  const fetchUserData = async () => {
    if (!token) return;
    try {
      const response = await fetch(`${API_BASE_URL}/users/me`, { headers: getAuthHeaders() });
      if (!response.ok) throw new Error('Failed to fetch user data');
      const userData = await response.json();
      setUserRole(userData.role);
      setUsername(userData.username);
      localStorage.setItem('userRole', userData.role);
      localStorage.setItem('username', userData.username);
    } catch (error) {
      console.error('Error fetching user data:', error);
      handleLogout();
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userRole');
    localStorage.removeItem('username');
    setToken(null);
    setUserRole(null);
    setUsername(null);
    setAnalytics(null);
    setFeedback(null);
    setLoading(false);
  };

  const fetchData = async () => {
    if (!token) return;
    setLoading(true);
    try {
      const headers = getAuthHeaders();

      const feedbackResponse = await fetch(`${API_BASE_URL}/feedback`, { headers });
      if (feedbackResponse.ok) {
        setFeedback(await feedbackResponse.json());
      }

      const storedRole = localStorage.getItem('userRole');
      if (storedRole === UserRole.ADMIN) {
        const analyticsResponse = await fetch(`${API_BASE_URL}/analytics`, { headers });
        if (analyticsResponse.ok) {
          setAnalytics(await analyticsResponse.json());
        }
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedbackSubmit = async (feedbackText: string) => {
    if (!token || !username) return;
    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ feedback: feedbackText, username }),
      });
      if (!response.ok) throw new Error('Failed to submit feedback');
      fetchData();
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  const LoadingSpinner = () => (
    <Box display="flex" justifyContent="center" my={4}>
      <CircularProgress size={60} thickness={4} />
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box display="flex" flexDirection="column" minHeight="100vh">
          <Header token={token} handleLogout={handleLogout} userRole={userRole} />
          <Container component="main" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
            <Routes>
              <Route path="/register" element={<Register />} />
              <Route path="/login" element={<Login setToken={setToken} />} />
              <Route path="/home" element={
                <PrivateRoute><Home /></PrivateRoute>
              } />
              <Route path="/analytics" element={
                <PrivateRoute requiredRole={UserRole.ADMIN}>
                  {loading ? <LoadingSpinner /> : <AnalyticsDisplay analytics={analytics} userRole={userRole} />}
                </PrivateRoute>
              } />
              <Route path="/feedback" element={
                <PrivateRoute>
                  <>
                    {userRole !== UserRole.ADMIN && <FeedbackForm onSubmit={handleFeedbackSubmit} />}
                    {loading ? <LoadingSpinner /> : <FeedbackDisplay feedback={feedback} userRole={userRole} username={username} />}
                  </>
                </PrivateRoute>
              } />
              <Route path="/" element={token ? <Navigate to="/home" /> : <Navigate to="/login" />} />
            </Routes>
          </Container>
          <Footer />
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;
