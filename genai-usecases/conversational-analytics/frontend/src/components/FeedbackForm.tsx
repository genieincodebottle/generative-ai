import React, { useState } from 'react';
import { TextField, Button, Box, Paper, Typography, Zoom, Container, Snackbar, IconButton } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import FeedbackIcon from '@mui/icons-material/Feedback';
import CloseIcon from '@mui/icons-material/Close';
import { keyframes } from '@emotion/react';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
`;

interface FeedbackFormProps {
  onSubmit: (feedback: string) => void;
}

const FeedbackForm: React.FC<FeedbackFormProps> = ({ onSubmit }) => {
  const [feedback, setFeedback] = useState('');
  const [openSnackbar, setOpenSnackbar] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(feedback);
    setFeedback('');
    setOpenSnackbar(true);
  };

  const handleCloseSnackbar = (event: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpenSnackbar(false);
  };

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
            Submit Your Feedback
          </Typography>
        </Box>

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
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              multiline
              rows={4}
              variant="outlined"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Enter your feedback here"
              required
              margin="normal"
              sx={{ mb: 2 }}
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              endIcon={<SendIcon />}
              sx={{ 
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                transition: 'all 0.3s',
                '&:hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 6px 10px rgba(0, 0, 0, 0.2)',
                }
              }}
            >
              Submit Feedback
            </Button>
          </form>
        </Paper>

        <Snackbar
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'center',
          }}
          open={openSnackbar}
          autoHideDuration={6000}
          onClose={handleCloseSnackbar}
          message="Feedback submitted successfully!"
          action={
            <IconButton
              size="small"
              aria-label="close"
              color="inherit"
              onClick={handleCloseSnackbar}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          }
        />
      </Container>
    </Zoom>
  );
};

export default FeedbackForm;