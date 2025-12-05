import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  TextField,
  Typography,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
  Chip,
} from '@mui/material';
import { Shield } from '@mui/icons-material';
import { authAPI } from '../../services/api';
import { useAuthStore } from '../../store/authStore';

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuthStore();
  const [tabValue, setTabValue] = useState(0); // 0 = login, 1 = register
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });
  const [registerData, setRegisterData] = useState({
    username: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    email: '',
    phone: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError(null);
  };

  const handleRegisterChange = (e) => {
    setRegisterData({
      ...registerData,
      [e.target.name]: e.target.value,
    });
    setError(null);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setError(null);
    setSuccess(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await authAPI.login(formData.username, formData.password);
      login(data.user, data.token);
      // Redirect based on role
      if (data.user.role === 'user') {
        navigate('/community');
      } else {
        navigate('/community');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    // Validation
    if (registerData.password !== registerData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    if (registerData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      setLoading(false);
      return;
    }

    try {
      await authAPI.register({
        username: registerData.username,
        password: registerData.password,
        full_name: registerData.full_name,
        email: registerData.email || null,
        phone: registerData.phone || null,
      });

      setSuccess('Registration successful! Please login with your credentials.');
      setRegisterData({
        username: '',
        password: '',
        confirmPassword: '',
        full_name: '',
        email: '',
        phone: '',
      });

      // Switch to login tab after 2 seconds
      setTimeout(() => {
        setTabValue(0);
        setSuccess(null);
      }, 2000);

    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        background: 'linear-gradient(135deg, #4BA3C7 0%, #5DADE2 50%, #85C1E2 100%)',
        py: 4,
      }}
    >
      <Container maxWidth="md">
        <Card elevation={6}>
          <CardContent sx={{ p: 4 }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Shield sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h4" gutterBottom>
                SafeGuard AI
              </Typography>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Content Moderation Platform
              </Typography>
              <Typography variant="body2" color="text.secondary">
                AI-Powered Multi-Agent System with Human-in-the-Loop
              </Typography>
            </Box>

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            {success && (
              <Alert severity="success" sx={{ mb: 3 }}>
                {success}
              </Alert>
            )}

            <Box sx={{ maxWidth: 480, mx: 'auto' }}>
              <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }} variant="fullWidth">
                <Tab label="Sign In" />
                <Tab label="Register" />
              </Tabs>

                {tabValue === 0 ? (
                  <form onSubmit={handleSubmit}>
                    <TextField
                      fullWidth
                      label="Username"
                      name="username"
                      value={formData.username}
                      onChange={handleChange}
                      margin="normal"
                      required
                      autoFocus
                      size="small"
                    />
                    <TextField
                      fullWidth
                      label="Password"
                      name="password"
                      type="password"
                      value={formData.password}
                      onChange={handleChange}
                      margin="normal"
                      required
                      size="small"
                    />
                    <Button
                      fullWidth
                      type="submit"
                      variant="contained"
                      size="large"
                      disabled={loading}
                      sx={{ mt: 2 }}
                    >
                      {loading ? <CircularProgress size={24} /> : 'Sign In'}
                    </Button>
                  </form>
                ) : (
                  <form onSubmit={handleRegisterSubmit}>
                    <TextField
                      fullWidth
                      label="Username"
                      name="username"
                      value={registerData.username}
                      onChange={handleRegisterChange}
                      margin="normal"
                      required
                      autoFocus
                      size="small"
                      inputProps={{ minLength: 3, maxLength: 50 }}
                      helperText="3-50 characters"
                    />
                    <TextField
                      fullWidth
                      label="Full Name"
                      name="full_name"
                      value={registerData.full_name}
                      onChange={handleRegisterChange}
                      margin="normal"
                      required
                      size="small"
                    />
                    <TextField
                      fullWidth
                      label="Email (Optional)"
                      name="email"
                      type="email"
                      value={registerData.email}
                      onChange={handleRegisterChange}
                      margin="normal"
                      size="small"
                    />
                    <TextField
                      fullWidth
                      label="Phone (Optional)"
                      name="phone"
                      value={registerData.phone}
                      onChange={handleRegisterChange}
                      margin="normal"
                      size="small"
                    />
                    <TextField
                      fullWidth
                      label="Password"
                      name="password"
                      type="password"
                      value={registerData.password}
                      onChange={handleRegisterChange}
                      margin="normal"
                      required
                      size="small"
                      inputProps={{ minLength: 6 }}
                      helperText="Minimum 6 characters"
                    />
                    <TextField
                      fullWidth
                      label="Confirm Password"
                      name="confirmPassword"
                      type="password"
                      value={registerData.confirmPassword}
                      onChange={handleRegisterChange}
                      margin="normal"
                      required
                      size="small"
                    />
                    <Button
                      fullWidth
                      type="submit"
                      variant="contained"
                      size="large"
                      disabled={loading}
                      sx={{ mt: 2 }}
                    >
                      {loading ? <CircularProgress size={24} /> : 'Register'}
                    </Button>
                  </form>
                )}
            </Box>

            <Divider sx={{ my: 4 }} />

            {/* System Flow */}
            <Box sx={{ bgcolor: 'grey.50', p: 3, borderRadius: 2 }}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ mb: 2 }}>
                Content Moderation Flow
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', justifyContent: 'center' }}>
                <Chip label="1. User Submits" size="small" />
                <Typography variant="caption" color="text.secondary">→</Typography>
                <Chip label="2. AI Analysis" size="small" color="primary" />
                <Typography variant="caption" color="text.secondary">→</Typography>
                <Chip label="3. Human Review" size="small" color="warning" />
                <Typography variant="caption" color="text.secondary">→</Typography>
                <Chip label="4. Decision" size="small" color="success" />
                <Typography variant="caption" color="text.secondary">→</Typography>
                <Chip label="5. Appeal?" size="small" color="info" />
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
}
