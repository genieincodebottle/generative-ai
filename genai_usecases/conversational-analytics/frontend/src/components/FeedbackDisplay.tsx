import React from 'react';
import { Box, Paper, Typography, List, ListItem, ListItemText, Zoom, Grid, Container } from '@mui/material';
import { keyframes } from '@emotion/react';
import FeedbackIcon from '@mui/icons-material/Feedback';
import { UserRole } from '../types';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
`;

interface FeedbackItem {
  feedback: string;
  timestamp: string;
  username: string;
}

interface FeedbackDisplayProps {
  feedback: {
    current: FeedbackItem[];
    historical: FeedbackItem[];
  } | null;
  userRole: UserRole | null;
  username: string | null;
}

const FeedbackDisplay: React.FC<FeedbackDisplayProps> = ({ feedback, userRole, username }) => {
  if (!feedback) return null;

  const filterFeedback = (feedbackList: FeedbackItem[]) => {
    if (userRole === UserRole.ADMIN) {
      return feedbackList;
    } else {
      return feedbackList.filter(item => item.username === username);
    }
  };

  const currentFeedback = filterFeedback(feedback.current);
  const historicalFeedback = filterFeedback(feedback.historical);

  return (
    <Zoom in={true} style={{ transitionDelay: '300ms' }}>
      <Container maxWidth="lg" sx={{ mt: 4, animation: `${fadeIn} 0.6s ease-out` }}>
        <Box 
          sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            mb: 4, 
            p: 2, 
            backgroundColor: '#2196f3', 
            borderRadius: '10px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
          }}
        >
          <FeedbackIcon sx={{ fontSize: 40, color: 'white', mr: 2 }} />
          <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'white' }}>
            Feedback Overview
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
                Current Feedback
              </Typography>
              <List>
                {currentFeedback.map((item, index) => (
                  <ListItem key={index}>
                    <ListItemText 
                      primary={<Typography variant="body1" sx={{ fontWeight: 'normal' }}>{item.feedback}</Typography>}
                      secondary={
                        <>
                          {userRole === UserRole.ADMIN && `${item.username} - `}
                          {new Date(item.timestamp).toLocaleString()}
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
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
                Historical Feedback
              </Typography>
              <List>
                {historicalFeedback.map((item, index) => (
                  <ListItem key={index}>
                    <ListItemText 
                      primary={<Typography variant="body1" sx={{ fontWeight: 'normal' }}>{item.feedback}</Typography>}
                      secondary={
                        <>
                          {userRole === UserRole.ADMIN && `${item.username} - `}
                          {new Date(item.timestamp).toLocaleString()}
                        </>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Zoom>
  );
};

export default FeedbackDisplay;