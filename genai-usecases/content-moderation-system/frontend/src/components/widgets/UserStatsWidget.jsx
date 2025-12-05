import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Divider,
  Grid
} from '@mui/material';

const StatItem = ({ value, label, color = 'primary' }) => (
  <Box sx={{ textAlign: 'center' }}>
    <Typography
      variant="h3"
      component="div"
      fontWeight="bold"
      color={`${color}.main`}
    >
      {value}
    </Typography>
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
  </Box>
);

const UserStatsWidget = ({ stats }) => {
  const {
    totalStories = 0,
    published = 0,
    underReview = 0,
    totalViews = 0
  } = stats;

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h5" component="h2" fontWeight="bold" gutterBottom>
          Your Stats
        </Typography>
        <Divider sx={{ mb: 3 }} />

        <Grid container spacing={3}>
          <Grid item xs={6} sm={3}>
            <StatItem value={totalStories} label="Total Stories" color="primary" />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatItem value={published} label="Published" color="success" />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatItem value={underReview} label="Under Review" color="warning" />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatItem value={totalViews} label="Total Views" color="info" />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default UserStatsWidget;
