import {
  Article,
  Cancel,
  CheckCircle,
  Gavel,
  HourglassEmpty,
  Refresh,
  Send,
  Warning,
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
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { appealsAPI, contentAPI } from '../../services/api';
import { useAuthStore } from '../../store/authStore';

const statusConfig = {
  submitted: { color: 'info', icon: <HourglassEmpty />, label: 'Processing' },
  approved: { color: 'success', icon: <CheckCircle />, label: 'Approved' },
  removed: { color: 'error', icon: <Cancel />, label: 'Removed' },
  warned: { color: 'warning', icon: <Warning />, label: 'Warning Issued' },
  flagged: { color: 'warning', icon: <Warning />, label: 'Under Review' },
  pending_human_review: { color: 'warning', icon: <HourglassEmpty />, label: 'Under Review' },
};

export default function UserPortal() {
  const { user } = useAuthStore();
  const [content, setContent] = useState('');
  const [platform, setPlatform] = useState('forum');
  const [contentType, setContentType] = useState('post');
  const [submitting, setSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const [myContent, setMyContent] = useState([]);
  const [loadingContent, setLoadingContent] = useState(true);

  // Appeal dialog state
  const [appealDialog, setAppealDialog] = useState({ open: false, contentId: null, contentText: '' });
  const [appealReason, setAppealReason] = useState('');
  const [submittingAppeal, setSubmittingAppeal] = useState(false);

  useEffect(() => {
    loadMyContent();
  }, []);

  const loadMyContent = async () => {
    setLoadingContent(true);
    try {
      const response = await contentAPI.getAllContent(null, 100);
      // Filter content by current user
      const userContent = response.content.filter(
        c => c.user_id === user?.user_id?.toString() || c.username === user?.username
      );
      setMyContent(userContent);
    } catch (err) {
      console.error('Error loading content:', err);
    } finally {
      setLoadingContent(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!content.trim()) return;

    setSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(false);

    try {
      await contentAPI.submitContent({
        content_text: content,
        content_type: contentType,
        platform: platform,
        user_id: user?.user_id?.toString(),
        username: user?.username,
        // Default user profile for demo
        account_age_days: 30,
        total_posts: myContent.length,
        reputation_score: 0.7,
      });

      setSubmitSuccess(true);
      setContent('');
      loadMyContent();

      setTimeout(() => setSubmitSuccess(false), 5000);
    } catch (err) {
      setSubmitError(err.response?.data?.detail || 'Failed to submit content');
    } finally {
      setSubmitting(false);
    }
  };

  const handleAppealOpen = (contentItem) => {
    setAppealDialog({
      open: true,
      contentId: contentItem.content_id,
      contentText: contentItem.content_text,
    });
    setAppealReason('');
  };

  const handleAppealSubmit = async () => {
    if (!appealReason.trim()) return;

    setSubmittingAppeal(true);
    try {
      await appealsAPI.submitAppeal({
        content_id: appealDialog.contentId,
        user_id: user?.user_id?.toString(),
        appeal_reason: appealReason,
      });

      setAppealDialog({ open: false, contentId: null, contentText: '' });
      loadMyContent();
    } catch (err) {
      console.error('Error submitting appeal:', err);
    } finally {
      setSubmittingAppeal(false);
    }
  };

  const getStatusChip = (status) => {
    const config = statusConfig[status] || statusConfig.submitted;
    return (
      <Chip
        icon={config.icon}
        label={config.label}
        color={config.color}
        size="small"
      />
    );
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="body1" color="text.secondary">
          Share your thoughts with the community. All content is reviewed by our AI moderation system to ensure a safe environment.
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Content Submission Form */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Article /> Create New Post
              </Typography>

              {submitSuccess && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  Content submitted successfully! It will be reviewed by our moderation system.
                </Alert>
              )}

              {submitError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {submitError}
                </Alert>
              )}

              <form onSubmit={handleSubmit}>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  label="What's on your mind?"
                  placeholder="Share your thoughts, questions, or ideas..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  sx={{ mb: 2 }}
                />

                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <FormControl fullWidth size="small">
                      <InputLabel>Platform</InputLabel>
                      <Select
                        value={platform}
                        label="Platform"
                        onChange={(e) => setPlatform(e.target.value)}
                      >
                        <MenuItem value="forum">Forum Discussion</MenuItem>
                        <MenuItem value="comment">Comment</MenuItem>
                        <MenuItem value="post">Social Post</MenuItem>
                        <MenuItem value="message">Direct Message</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={6}>
                    <FormControl fullWidth size="small">
                      <InputLabel>Content Type</InputLabel>
                      <Select
                        value={contentType}
                        label="Content Type"
                        onChange={(e) => setContentType(e.target.value)}
                      >
                        <MenuItem value="post">Text Post</MenuItem>
                        <MenuItem value="comment">Comment</MenuItem>
                        <MenuItem value="reply">Reply</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>

                <Button
                  type="submit"
                  variant="contained"
                  startIcon={submitting ? <CircularProgress size={20} color="inherit" /> : <Send />}
                  disabled={submitting || !content.trim()}
                >
                  {submitting ? 'Submitting...' : 'Submit Content'}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* My Content History */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  My Content History
                </Typography>
                <Button
                  size="small"
                  startIcon={<Refresh />}
                  onClick={loadMyContent}
                  disabled={loadingContent}
                >
                  Refresh
                </Button>
              </Box>

              {loadingContent ? (
                <LinearProgress />
              ) : myContent.length === 0 ? (
                <Alert severity="info">
                  You haven't submitted any content yet. Start by creating a post above!
                </Alert>
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Content</TableCell>
                        <TableCell>Platform</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Date</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {myContent.map((item) => (
                        <TableRow key={item.content_id}>
                          <TableCell sx={{ maxWidth: 300 }}>
                            <Typography variant="body2" noWrap>
                              {item.content_text}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={item.platform} size="small" variant="outlined" />
                          </TableCell>
                          <TableCell>
                            {getStatusChip(item.status || item.current_status)}
                          </TableCell>
                          <TableCell>
                            {new Date(item.created_at || item.submission_timestamp).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            {(item.status === 'removed' || item.status === 'warned' ||
                              item.current_status === 'removed' || item.current_status === 'warned') && (
                              <Button
                                size="small"
                                startIcon={<Gavel />}
                                onClick={() => handleAppealOpen(item)}
                              >
                                Appeal
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Sidebar - Guidelines & Stats */}
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

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                How Moderation Works
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip label="1" size="small" color="primary" />
                  <Typography variant="body2">You submit content</Typography>
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

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Your Stats
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="h4" color="primary">
                    {myContent.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Posts
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="info.main">
                    {myContent.filter(c =>
                      ['flagged', 'pending_human_review', 'submitted', 'under_review', 'analyzing'].includes(c.status || c.current_status)
                    ).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Under Review
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="success.main">
                    {myContent.filter(c => c.status === 'approved' || c.current_status === 'approved').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Approved
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="warning.main">
                    {myContent.filter(c => c.status === 'warned' || c.current_status === 'warned').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Warnings
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="h4" color="error.main">
                    {myContent.filter(c => c.status === 'removed' || c.current_status === 'removed').length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Removed
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Appeal Dialog */}
      <Dialog open={appealDialog.open} onClose={() => setAppealDialog({ open: false, contentId: null, contentText: '' })} maxWidth="sm" fullWidth>
        <DialogTitle>Appeal Content Decision</DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            If you believe your content was incorrectly moderated, please explain why.
          </Alert>

          <Typography variant="subtitle2" gutterBottom>
            Original Content:
          </Typography>
          <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
            <Typography variant="body2">
              {appealDialog.contentText}
            </Typography>
          </Paper>

          <TextField
            fullWidth
            multiline
            rows={4}
            label="Reason for Appeal"
            placeholder="Explain why you believe this decision should be reconsidered..."
            value={appealReason}
            onChange={(e) => setAppealReason(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAppealDialog({ open: false, contentId: null, contentText: '' })}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleAppealSubmit}
            disabled={!appealReason.trim() || submittingAppeal}
          >
            {submittingAppeal ? 'Submitting...' : 'Submit Appeal'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
