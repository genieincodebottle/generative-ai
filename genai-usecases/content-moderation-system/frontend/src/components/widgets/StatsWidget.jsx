import React from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import {
  Article as ArticleIcon,
  Visibility as VisibilityIcon,
  Comment as CommentIcon,
  Person as PersonIcon
} from '@mui/icons-material';

const StatCard = ({ title, value, icon, color = 'primary' }) => (
  <Paper
    elevation={0}
    sx={{
      p: 2.5,
      bgcolor: `${color}.light`,
      color: `${color}.contrastText`,
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between'
    }}
  >
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <Box>
        <Typography variant="h3" fontWeight="bold" sx={{ mb: 0.5 }}>
          {value?.toLocaleString() || 0}
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.9 }}>
          {title}
        </Typography>
      </Box>
      <Box sx={{ opacity: 0.7 }}>
        {icon}
      </Box>
    </Box>
  </Paper>
);

const StatsWidget = ({ stats = {} }) => {
  const {
    totalStories = 0,
    totalViews = 0,
    totalComments = 0,
    myStories = 0
  } = stats;

  const statItems = [
    {
      title: 'Total Stories',
      value: totalStories,
      icon: <ArticleIcon sx={{ fontSize: 40 }} />,
      color: 'primary'
    },
    {
      title: 'Total Views',
      value: totalViews,
      icon: <VisibilityIcon sx={{ fontSize: 40 }} />,
      color: 'success'
    },
    {
      title: 'Total Comments',
      value: totalComments,
      icon: <CommentIcon sx={{ fontSize: 40 }} />,
      color: 'info'
    },
    {
      title: 'My Stories',
      value: myStories,
      icon: <PersonIcon sx={{ fontSize: 40 }} />,
      color: 'secondary'
    }
  ];

  return (
    <Grid container spacing={2}>
      {statItems.map((stat, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <StatCard {...stat} />
        </Grid>
      ))}
    </Grid>
  );
};

export default StatsWidget;
