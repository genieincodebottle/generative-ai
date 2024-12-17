import React, { useState, useEffect } from 'react';
import { Typography, Box, Card, CardContent, Avatar, Chip, Grid } from '@mui/material';
import { styled } from '@mui/system';
import { UserRole } from '../types';
import PersonIcon from '@mui/icons-material/Person';
import WorkIcon from '@mui/icons-material/Work';
import EmailIcon from '@mui/icons-material/Email';
import DateRangeIcon from '@mui/icons-material/DateRange';

interface UserData {
  username: string;
  role: UserRole;
  email: string;
  joinDate: string;
}

const PageContainer = styled(Box)(({ theme }) => ({
  //background: 'linear-gradient(120deg, #e0f7fa 0%, #b2ebf2 100%)',
  minHeight: 'calc(100vh - 64px - 60px)', // Adjust for header and footer
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  padding: theme.spacing(3),
}));

const StyledCard = styled(Card)(({ theme }) => ({
  maxWidth: 600,
  width: '100%',
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: '0 8px 40px rgba(0, 0, 0, 0.12)',
  overflow: 'hidden',
}));

const CardHeader = styled(Box)(({ theme }) => ({
  background: 'linear-gradient(60deg, #1eb6f7 0%, #05a6eb 100%)',
  color: theme.palette.common.white,
  padding: theme.spacing(3),
  textAlign: 'center',
}));

const AvatarWrapper = styled(Box)(({ theme }) => ({
  marginTop: theme.spacing(-8),
  marginBottom: theme.spacing(2),
}));

const LargeAvatar = styled(Avatar)(({ theme }) => ({
  width: theme.spacing(16),
  height: theme.spacing(16),
  margin: '0 auto',
  border: `4px solid ${theme.palette.background.paper}`,
  boxShadow: '0 3px 5px 2px rgba(0, 0, 0, 0.3)',
}));

const InfoItem = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(2),
}));

const Home: React.FC = () => {
  const [userData, setUserData] = useState<UserData | null>(null);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('No token found');
        }

        const response = await fetch('http://localhost:8000/users/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!response.ok) throw new Error('Failed to fetch user data');
        const data = await response.json();
        setUserData({
          ...data
        });
      } catch (error) {
        console.error('Error fetching user data:', error);
      }
    };

    fetchUserData();
  }, []);

  if (!userData) return <Typography>Loading...</Typography>;

  return (
    <PageContainer>
      <StyledCard>
        <CardHeader>
          <AvatarWrapper>
            <LargeAvatar alt={userData.username}>
              {userData.username.charAt(0).toUpperCase()}
            </LargeAvatar>
          </AvatarWrapper>
          <Typography variant="h4" gutterBottom>
            Welcome, {userData.username}!
          </Typography>
          <Chip
            label={userData.role}
            color="secondary"
            icon={<WorkIcon />}
          />
        </CardHeader>
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <InfoItem>
                <PersonIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Typography variant="body1">
                  Username: {userData.username}
                </Typography>
              </InfoItem>
            </Grid>
            <Grid item xs={12}>
              <InfoItem>
                <WorkIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Typography variant="body1">
                  Role: {userData.role}
                </Typography>
              </InfoItem>
            </Grid>
          </Grid>
        </CardContent>
      </StyledCard>
    </PageContainer>
  );
}

export default Home;