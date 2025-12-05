import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Avatar,
  Chip,
  Divider,
  Button,
  Stack,
  Paper
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Comment as CommentIcon
} from '@mui/icons-material';

const StoryCard = ({ story, onClick, showStatus = false }) => {
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

  const truncateText = (text, maxLength = 150) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'approved':
      case 'published':
      case 'human_approved':
        return 'success';
      case 'pending':
      case 'under review':
      case 'pending_human_review':
      case 'flagged':
      case 'warned':
        return 'warning';
      case 'escalated':
      case 'removed':
      case 'rejected':
      case 'human_rejected':
        return 'error';
      case 'human_review_completed':
        return 'info';
      default:
        return 'default';
    }
  };

  const getStatusLabel = (status) => {
    const statusLower = status?.toLowerCase();
    const labelMap = {
      pending: 'Under Review',
      approved: 'Published',
      removed: 'Removed',
      flagged: 'Flagged',
      pending_human_review: 'Awaiting HITL',
      human_review_completed: 'Human Reviewed',
      escalated: 'Escalated',
      warned: 'Warned',
      human_approved: 'Human Approved',
      human_rejected: 'Human Rejected',
    };
    return labelMap[statusLower] || status;
  };

  return (
    <Card
      elevation={2}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        cursor: 'pointer',
        transition: 'all 0.2s',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: 4
        }
      }}
      onClick={() => onClick && onClick(story.story_id)}
    >
      <CardContent sx={{ flexGrow: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ flex: 1, pr: 1 }}>
            {story.title}
          </Typography>
          {showStatus && story.moderation_status && (
            <Chip
              label={getStatusLabel(story.moderation_status)}
              size="small"
              color={getStatusColor(story.moderation_status)}
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <Avatar sx={{ width: 32, height: 32 }}>
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

        <Typography variant="body2" color="text.secondary">
          {truncateText(story.content_text)}
        </Typography>
      </CardContent>

      <Divider />

      <CardActions sx={{ justifyContent: 'space-between', px: 2, py: 1.5 }}>
        <Stack direction="row" spacing={2}>
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
        </Stack>
        <Button size="small">Read More</Button>
      </CardActions>
    </Card>
  );
};

const StoriesGrid = ({ stories = [], onStoryClick, showStatus = false, emptyMessage = 'No stories yet' }) => {
  if (stories.length === 0) {
    return (
      <Paper elevation={0} sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
        <Typography variant="h6" color="text.secondary">
          {emptyMessage}
        </Typography>
      </Paper>
    );
  }

  return (
    <Grid container spacing={3}>
      {stories.map((story) => (
        <Grid item xs={12} sm={6} lg={4} key={story.story_id}>
          <StoryCard story={story} onClick={onStoryClick} showStatus={showStatus} />
        </Grid>
      ))}
    </Grid>
  );
};

export default StoriesGrid;
