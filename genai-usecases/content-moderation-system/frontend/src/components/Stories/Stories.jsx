import {
  Add,
  ChatBubbleOutline,
  CheckCircle,
  HourglassEmpty,
  Refresh,
  Visibility,
  Warning,
  Flag,
  TrendingUp,
  VerifiedUser,
  Cancel,
} from '@mui/icons-material';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Grid,
  LinearProgress,
  TextField,
  Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { storiesAPI } from '../../services/api';
import { useAuthStore } from '../../store/authStore';

const statusConfig = {
  approved: { color: 'success', icon: <CheckCircle fontSize="small" />, label: 'Published' },
  pending: { color: 'warning', icon: <HourglassEmpty fontSize="small" />, label: 'Under Review' },
  pending_human_review: { color: 'warning', icon: <Warning fontSize="small" />, label: 'Awaiting HITL' },
  human_review_completed: { color: 'info', icon: <VerifiedUser fontSize="small" />, label: 'Human Reviewed' },
  flagged: { color: 'warning', icon: <Flag fontSize="small" />, label: 'Flagged' },
  escalated: { color: 'error', icon: <TrendingUp fontSize="small" />, label: 'Escalated' },
  warned: { color: 'warning', icon: <Flag fontSize="small" />, label: 'Warned' },
  removed: { color: 'error', icon: <Cancel fontSize="small" />, label: 'Removed' },
  human_approved: { color: 'success', icon: <CheckCircle fontSize="small" />, label: 'Human Approved' },
  human_rejected: { color: 'error', icon: <Cancel fontSize="small" />, label: 'Human Rejected' },
};

export default function Stories() {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const [stories, setStories] = useState([]);
  const [myStories, setMyStories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('all'); // 'all' or 'mine'

  // New story dialog
  const [storyDialog, setStoryDialog] = useState(false);
  const [storyTitle, setStoryTitle] = useState('');
  const [storyContent, setStoryContent] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(null);
  const [submitError, setSubmitError] = useState(null);

  useEffect(() => {
    loadStories();
  }, []);

  // Reload stories when page becomes visible (tab switch)
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadStories();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Also reload when window gains focus
    window.addEventListener('focus', loadStories);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', loadStories);
    };
  }, []);

  const loadStories = async () => {
    setLoading(true);
    try {
      // Load all visible stories
      const response = await storiesAPI.getStories(true, 100);
      setStories(response.stories || []);

      // Load user's own stories (including pending)
      if (user?.user_id) {
        const userResponse = await storiesAPI.getUserStories(user.user_id.toString(), 100);
        setMyStories(userResponse.stories || []);
      }
    } catch (err) {
      console.error('Error loading stories:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitStory = async () => {
    if (!storyTitle.trim() || !storyContent.trim()) return;

    setSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(null);

    try {
      const result = await storiesAPI.submitStory({
        title: storyTitle,
        content_text: storyContent,
        user_id: user?.user_id?.toString(),
        username: user?.username,
      });

      if (result.is_approved) {
        setSubmitSuccess('Your story has been published!');
        setActiveTab('all'); // Switch to All Stories to see published story
      } else {
        setSubmitSuccess('Your story has been submitted and is under review.');
        setActiveTab('mine'); // Switch to My Stories to see pending story
      }

      setStoryTitle('');
      setStoryContent('');
      setStoryDialog(false);

      // Wait a moment for backend to complete, then refresh
      setTimeout(() => {
        loadStories();
      }, 500);

      setTimeout(() => setSubmitSuccess(null), 5000);
    } catch (err) {
      setSubmitError(err.response?.data?.detail || 'Failed to submit story');
    } finally {
      setSubmitting(false);
    }
  };

  const getDisplayedStories = () => {
    if (activeTab === 'mine') {
      return myStories;
    }
    return stories;
  };

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const truncateText = (text, maxLength = 200) => {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Community Stories
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Share your experiences and read stories from the community
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            startIcon={<Refresh />}
            onClick={loadStories}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setStoryDialog(true)}
          >
            Write Story
          </Button>
        </Box>
      </Box>

      {/* Success/Error Alerts */}
      {submitSuccess && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSubmitSuccess(null)}>
          {submitSuccess}
        </Alert>
      )}
      {submitError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setSubmitError(null)}>
          {submitError}
        </Alert>
      )}

      {/* Tabs */}
      <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
        <Chip
          label="All Stories"
          onClick={() => setActiveTab('all')}
          color={activeTab === 'all' ? 'primary' : 'default'}
          variant={activeTab === 'all' ? 'filled' : 'outlined'}
        />
        <Chip
          label="My Stories"
          onClick={() => setActiveTab('mine')}
          color={activeTab === 'mine' ? 'primary' : 'default'}
          variant={activeTab === 'mine' ? 'filled' : 'outlined'}
        />
      </Box>

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Grid container spacing={3}>
        {/* Main Content - Stories */}
        <Grid item xs={12} md={8}>
          {/* Stories List */}
          {!loading && getDisplayedStories().length === 0 ? (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  {activeTab === 'mine' ? 'You haven\'t written any stories yet' : 'No stories yet'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Be the first to share your story with the community!
                </Typography>
                <Button variant="contained" startIcon={<Add />} onClick={() => setStoryDialog(true)}>
                  Write Your First Story
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Grid container spacing={2}>
              {getDisplayedStories().map((story) => {
                const status = statusConfig[story.moderation_status] || statusConfig.pending;
                const isVisible = story.is_visible;

                return (
                  <Grid item xs={12} key={story.story_id}>
                    <Card
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        cursor: isVisible ? 'pointer' : 'default',
                        opacity: isVisible ? 1 : 0.7,
                        '&:hover': isVisible ? {
                          boxShadow: 4,
                          transform: 'translateY(-2px)',
                          transition: 'all 0.2s ease-in-out',
                        } : {},
                      }}
                      onClick={() => isVisible && navigate(`/stories/${story.story_id}`)}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                          <Typography variant="h6" component="h2" sx={{ fontWeight: 600 }}>
                            {story.title}
                          </Typography>
                          {activeTab === 'mine' && (
                            <Chip
                              icon={status.icon}
                              label={status.label}
                              color={status.color}
                              size="small"
                            />
                          )}
                        </Box>

                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          By {story.username} • {formatDate(story.created_at)}
                        </Typography>

                        <Typography variant="body1" color="text.primary" sx={{ mb: 2 }}>
                          {truncateText(story.content_text)}
                        </Typography>

                        <Divider sx={{ my: 2 }} />

                        <Box sx={{ display: 'flex', gap: 2, color: 'text.secondary' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Visibility fontSize="small" />
                            <Typography variant="body2">{story.view_count || 0}</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <ChatBubbleOutline fontSize="small" />
                            <Typography variant="body2">{story.comment_count || 0}</Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          )}
        </Grid>

        {/* Sidebar - Guidelines & Info */}
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Community Guidelines
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Typography variant="body2" paragraph>
                <strong>Be Respectful:</strong> Treat others with kindness and respect.
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>No Hate Speech:</strong> Discrimination based on race, gender, religion, etc. is not allowed.
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>No Harassment:</strong> Bullying, threats, or targeted harassment will result in removal.
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Stay On Topic:</strong> Keep discussions relevant and constructive.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Violations may result in content removal, warnings, or account suspension.
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                How Moderation Works
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="1" size="small" color="primary" />
                  <Typography variant="body2">You submit a story or comment</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="2" size="small" color="primary" />
                  <Typography variant="body2">AI analyzes for policy violations</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="3" size="small" color="primary" />
                  <Typography variant="body2">Flagged content goes to human review</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="4" size="small" color="primary" />
                  <Typography variant="body2">Decision made (approve/warn/remove)</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="5" size="small" color="primary" />
                  <Typography variant="body2">You can appeal if you disagree</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* My Stats */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Your Stats
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="h4" color="primary">
                    {myStories.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Stories
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="success.main">
                    {myStories.filter(s => s.is_visible || s.moderation_status === 'approved').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Published
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="warning.main">
                    {myStories.filter(s => ['pending', 'pending_human_review', 'flagged'].includes(s.moderation_status)).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Under Review
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="info.main">
                    {myStories.reduce((acc, s) => acc + (s.view_count || 0), 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Views
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* New Story Dialog */}
      <Dialog open={storyDialog} onClose={() => setStoryDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Write a New Story</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            Your story will be reviewed by our moderation system before being published.
          </Alert>

          <TextField
            fullWidth
            label="Story Title"
            placeholder="Give your story a compelling title..."
            value={storyTitle}
            onChange={(e) => setStoryTitle(e.target.value)}
            sx={{ mb: 2, mt: 1 }}
          />

          <TextField
            fullWidth
            multiline
            rows={10}
            label="Your Story"
            placeholder="Share your story with the community..."
            value={storyContent}
            onChange={(e) => setStoryContent(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStoryDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSubmitStory}
            disabled={!storyTitle.trim() || !storyContent.trim() || submitting}
            startIcon={submitting ? <CircularProgress size={20} color="inherit" /> : null}
          >
            {submitting ? 'Submitting...' : 'Submit Story'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
