import {
  ArrowBack,
  ChatBubbleOutline,
  CheckCircle,
  HourglassEmpty,
  Person,
  Send,
  Visibility,
  Warning,
} from '@mui/icons-material';
import {
  Alert,
  Avatar,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  LinearProgress,
  Paper,
  TextField,
  Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { storiesAPI } from '../../services/api';
import { useAuthStore } from '../../store/authStore';

const statusConfig = {
  approved: { color: 'success', icon: <CheckCircle fontSize="small" />, label: 'Approved' },
  pending: { color: 'warning', icon: <HourglassEmpty fontSize="small" />, label: 'Under Review' },
  pending_human_review: { color: 'warning', icon: <HourglassEmpty fontSize="small" />, label: 'Under Review' },
  removed: { color: 'error', icon: <Warning fontSize="small" />, label: 'Removed' },
};

export default function StoryDetail() {
  const { storyId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuthStore();

  const [story, setStory] = useState(null);
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Comment form state
  const [newComment, setNewComment] = useState('');
  const [submittingComment, setSubmittingComment] = useState(false);
  const [commentSuccess, setCommentSuccess] = useState(null);
  const [commentError, setCommentError] = useState(null);

  useEffect(() => {
    loadStory();
  }, [storyId]);

  const loadStory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await storiesAPI.getStoryById(storyId, true);
      setStory(response);
      setComments(response.comments || []);
    } catch (err) {
      console.error('Error loading story:', err);
      setError('Failed to load story. It may have been removed or is still under review.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitComment = async () => {
    if (!newComment.trim()) return;

    setSubmittingComment(true);
    setCommentError(null);
    setCommentSuccess(null);

    try {
      const result = await storiesAPI.submitComment(storyId, {
        content_text: newComment,
        user_id: user?.user_id?.toString(),
        username: user?.username,
      });

      if (result.is_approved) {
        setCommentSuccess('Your comment has been posted!');
        // Reload comments to show new one
        const updatedStory = await storiesAPI.getStoryById(storyId, true);
        setComments(updatedStory.comments || []);
      } else {
        setCommentSuccess('Your comment has been submitted and is under review.');
      }

      setNewComment('');
      setTimeout(() => setCommentSuccess(null), 5000);
    } catch (err) {
      setCommentError(err.response?.data?.detail || 'Failed to submit comment');
    } finally {
      setSubmittingComment(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getAvatarColor = (username) => {
    const colors = ['#1976d2', '#dc004e', '#2e7d32', '#ed6c02', '#0288d1', '#9c27b0'];
    const index = username ? username.charCodeAt(0) % colors.length : 0;
    return colors[index];
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <LinearProgress />
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          Loading story...
        </Typography>
      </Container>
    );
  }

  if (error || !story) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Button startIcon={<ArrowBack />} onClick={() => navigate('/stories')} sx={{ mb: 2 }}>
          Back to Stories
        </Button>
        <Alert severity="error">{error || 'Story not found'}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      {/* Back Button */}
      <Button startIcon={<ArrowBack />} onClick={() => navigate('/stories')} sx={{ mb: 3 }}>
        Back to Stories
      </Button>

      {/* Story Card */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          {/* Title */}
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
            {story.title}
          </Typography>

          {/* Author & Meta */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
            <Avatar sx={{ bgcolor: getAvatarColor(story.username) }}>
              {story.username?.charAt(0).toUpperCase() || <Person />}
            </Avatar>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                {story.username}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {formatDate(story.created_at)}
              </Typography>
            </Box>
          </Box>

          <Divider sx={{ mb: 3 }} />

          {/* Story Content */}
          <Typography
            variant="body1"
            sx={{
              whiteSpace: 'pre-wrap',
              lineHeight: 1.8,
              fontSize: '1.1rem',
            }}
          >
            {story.content_text}
          </Typography>

          <Divider sx={{ my: 3 }} />

          {/* Stats */}
          <Box sx={{ display: 'flex', gap: 3, color: 'text.secondary' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Visibility />
              <Typography variant="body2">{story.view_count || 0} views</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <ChatBubbleOutline />
              <Typography variant="body2">{comments.length} comments</Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Comments Section */}
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ChatBubbleOutline /> Comments ({comments.length})
      </Typography>

      {/* Comment Form */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          {commentSuccess && (
            <Alert severity="success" sx={{ mb: 2 }} onClose={() => setCommentSuccess(null)}>
              {commentSuccess}
            </Alert>
          )}
          {commentError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setCommentError(null)}>
              {commentError}
            </Alert>
          )}

          <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
            <Avatar sx={{ bgcolor: getAvatarColor(user?.username) }}>
              {user?.username?.charAt(0).toUpperCase() || <Person />}
            </Avatar>
            <Box sx={{ flexGrow: 1 }}>
              <TextField
                fullWidth
                multiline
                rows={3}
                placeholder="Write a comment..."
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                sx={{ mb: 1 }}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Comments are moderated for community safety
                </Typography>
                <Button
                  variant="contained"
                  endIcon={submittingComment ? <CircularProgress size={20} color="inherit" /> : <Send />}
                  onClick={handleSubmitComment}
                  disabled={!newComment.trim() || submittingComment}
                >
                  {submittingComment ? 'Posting...' : 'Post Comment'}
                </Button>
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Comments List */}
      {comments.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 4, textAlign: 'center' }}>
          <ChatBubbleOutline sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          <Typography variant="body1" color="text.secondary">
            No comments yet. Be the first to share your thoughts!
          </Typography>
        </Paper>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {comments.map((comment) => (
            <Card key={comment.comment_id} variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Avatar sx={{ bgcolor: getAvatarColor(comment.username), width: 36, height: 36 }}>
                    {comment.username?.charAt(0).toUpperCase() || <Person />}
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {comment.username}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(comment.created_at)}
                      </Typography>
                    </Box>
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                      {comment.content_text}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          ))}
        </Box>
      )}
    </Container>
  );
}
