import React from 'react';
import { Grid, Card, CardContent, CardActions, Typography, Box, Avatar, Chip, Divider } from '@mui/material';
import {
  Visibility as ViewIcon,
  Comment as CommentIcon,
  TrendingUp as TrendingIcon
} from '@mui/icons-material';

const FeaturedStoryCard = ({ story, onClick }) => {
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const truncateText = (text, maxLength = 120) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <Card
      elevation={3}
      sx={{
        height: '100%',
        cursor: 'pointer',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 6
        }
      }}
      onClick={() => onClick && onClick(story.story_id)}
    >
      <CardContent>
        <Typography variant="h6" fontWeight="bold" gutterBottom noWrap>
          {story.title}
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <Avatar sx={{ width: 28, height: 28, fontSize: '0.875rem' }}>
            {story.username?.charAt(0).toUpperCase()}
          </Avatar>
          <Box>
            <Typography variant="body2" fontWeight="medium">
              {story.username}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatDate(story.created_at)}
            </Typography>
          </Box>
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {truncateText(story.content_text)}
        </Typography>
      </CardContent>

      <Divider />

      <CardActions sx={{ justifyContent: 'space-between', px: 2, py: 1.5 }}>
        <Box sx={{ display: 'flex', gap: 1.5 }}>
          <Chip
            icon={<ViewIcon fontSize="small" />}
            label={story.view_count || 0}
            size="small"
            variant="outlined"
          />
          <Chip
            icon={<CommentIcon fontSize="small" />}
            label={story.comment_count || 0}
            size="small"
            variant="outlined"
            color="primary"
          />
        </Box>
      </CardActions>
    </Card>
  );
};

const FeaturedStoriesWidget = ({ stories = [], onStoryClick }) => {
  if (!stories || stories.length === 0) {
    return null;
  }

  return (
    <Box sx={{ mb: 4 }}>
      <Typography
        variant="h5"
        fontWeight="bold"
        gutterBottom
        sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}
      >
        <TrendingIcon color="primary" />
        Featured Stories
      </Typography>

      <Grid container spacing={3}>
        {stories.map((story) => (
          <Grid item xs={12} md={4} key={story.story_id}>
            <FeaturedStoryCard story={story} onClick={onStoryClick} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default FeaturedStoriesWidget;
