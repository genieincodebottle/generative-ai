import React from 'react';
import { Box, Paper, Typography, Chip, Zoom, Grid, Container } from '@mui/material';
import { Analytics, UserRole } from '../types';
import { keyframes } from '@emotion/react';
import BarChartIcon from '@mui/icons-material/BarChart';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
`;

interface AnalyticsDisplayProps {
  analytics: Partial<Analytics> | null;
  userRole: UserRole | null;
}

const AnalyticsDisplay: React.FC<AnalyticsDisplayProps> = ({ analytics, userRole }) => {
  if (!analytics || userRole !== UserRole.ADMIN) {
    return null;
  }

  return (
    <Zoom in={true} style={{ transitionDelay: '300ms' }}>
      <Container maxWidth="lg" sx={{ mt: 4, animation: `${fadeIn} 0.6s ease-out` }}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 4, 
            p: 2, 
            backgroundColor: '#05A6EB', 
            borderRadius: '10px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
          }}
        >
          <BarChartIcon sx={{ fontSize: 40, color: 'white', mr: 2 }} />
          <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'white' }}>
            Analytics Overview
          </Typography>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={6} 
              sx={{ 
                p: 3, 
                background: 'linear-gradient(145deg, #ffffff, #f0f0f0)', 
                borderRadius: '15px',
                transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-5px)',
                  boxShadow: '0 8px 15px rgba(0, 0, 0, 0.1)'
                }
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: 'black' }}>
                Key Topics
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
              {analytics.topics?.map((topic, index) => (
                <Chip
                  key={index}
                  label={topic}
                  sx={{
                    backgroundColor: 'white', 
                    color: 'black', 
                    fontWeight: 'bold',
                    transition: 'transform 0.2s',
                    '&:hover': {
                      transform: 'scale(1.05)'
                    }
                  }}
                />
              ))}
            </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={6} 
              sx={{ 
                p: 3, 
                background: 'linear-gradient(145deg, #ffffff, #f0f0f0)', 
                borderRadius: '15px',
                transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-5px)',
                  boxShadow: '0 8px 15px rgba(0, 0, 0, 0.1)'
                }
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: 'black' }}>
                Overall Sentiment
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
               {analytics.sentiment}
            </Box>
            </Paper>
          </Grid>
          <Grid item xs={12}>
            <Paper 
              elevation={6} 
              sx={{ 
                p: 3, 
                background: 'linear-gradient(145deg, #ffffff, #f0f0f0)', 
                borderRadius: '15px',
                transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-5px)',
                  boxShadow: '0 8px 15px rgba(0, 0, 0, 0.1)'
                }
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: 'black' }}>
                Emerging Trends
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
               {analytics.trends?.map((trend, index) => (
                <Chip
                  key={index}
                  label={trend}
                  sx={{
                    backgroundColor: 'white', 
                    color: 'black', 
                    fontWeight: 'bold',
                    transition: 'transform 0.2s',
                    '&:hover': {
                      transform: 'scale(1.05)'
                    }
                  }}
                />
              ))}
            </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Zoom>
  );
};

export default AnalyticsDisplay;