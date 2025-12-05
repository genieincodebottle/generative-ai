import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Grid,
  Paper,
  Typography,
  Alert,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ArrowBack,
  CheckCircle,
  Cancel,
  Warning,
  Person,
  CalendarToday,
  Language,
  ThumbUp,
  ThumbDown,
  Flag,
  VerifiedUser,
  TrendingUp,
} from '@mui/icons-material';
import { contentAPI } from '../../services/api';
import api from '../../services/api';

export default function ContentReview() {
  const { contentId } = useParams();
  const navigate = useNavigate();
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [moderating, setModerating] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Manual review dialog
  const [reviewDialog, setReviewDialog] = useState({ open: false, action: '' });
  const [reviewNotes, setReviewNotes] = useState('');
  const [reviewLoading, setReviewLoading] = useState(false);

  useEffect(() => {
    if (contentId) {
      fetchContent();
    }
  }, [contentId]);

  const fetchContent = async () => {
    try {
      setLoading(true);
      const data = await contentAPI.getContentById(contentId);
      setContent(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load content');
    } finally {
      setLoading(false);
    }
  };

  const handleModerate = async () => {
    try {
      setModerating(true);
      await contentAPI.moderateContent(contentId);
      await fetchContent(); // Refresh to see AI decision
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to moderate content');
    } finally {
      setModerating(false);
    }
  };

  const handleOpenReview = (action) => {
    setReviewDialog({ open: true, action });
    setReviewNotes('');
  };

  const handleCloseReview = () => {
    setReviewDialog({ open: false, action: '' });
    setReviewNotes('');
  };

  const handleSubmitReview = async () => {
    if (!reviewDialog.action) return;

    setReviewLoading(true);
    try {
      await api.post(`/api/content/${contentId}/manual-review`, {
        decision: reviewDialog.action,
        reviewer_name: 'Content Reviewer',
        notes: reviewNotes || `Manual ${reviewDialog.action} from review page`,
      });

      setSuccess(`Content ${reviewDialog.action}d successfully`);
      handleCloseReview();
      await fetchContent(); // Refresh to see updated status

      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err.response?.data?.detail || `Failed to ${reviewDialog.action} content`);
    } finally {
      setReviewLoading(false);
    }
  };

  const getToxicityColor = (score) => {
    if (score >= 0.7) return 'error';
    if (score >= 0.4) return 'warning';
    return 'success';
  };

  const getToxicityLabel = (score) => {
    if (score >= 0.7) return 'High Risk';
    if (score >= 0.4) return 'Medium Risk';
    return 'Low Risk';
  };

  const getStatusColor = (status) => {
    const statusMap = {
      pending: 'warning',
      approved: 'success',
      removed: 'error',
      flagged: 'warning',
      reviewed: 'info',
      pending_human_review: 'warning',
      human_review_completed: 'info',
      escalated: 'error',
      warned: 'warning',
      human_approved: 'success',
      human_rejected: 'error',
    };
    return statusMap[status?.toLowerCase()] || 'default';
  };

  const getStatusIcon = (status) => {
    const statusLower = status?.toLowerCase();
    if (statusLower === 'approved' || statusLower === 'human_approved') return <CheckCircle />;
    if (statusLower === 'removed' || statusLower === 'human_rejected') return <Cancel />;
    if (statusLower === 'flagged' || statusLower === 'pending_human_review') return <Warning />;
    if (statusLower === 'escalated') return <TrendingUp />;
    if (statusLower === 'human_review_completed') return <VerifiedUser />;
    if (statusLower === 'warned') return <Flag />;
    return null;
  };

  const getStatusLabel = (status) => {
    const statusLower = status?.toLowerCase();
    const labelMap = {
      pending: 'Pending',
      approved: 'Approved',
      removed: 'Removed',
      flagged: 'Flagged',
      reviewed: 'Reviewed',
      pending_human_review: 'Awaiting HITL',
      human_review_completed: 'Human Reviewed',
      escalated: 'Escalated',
      warned: 'Warned',
      human_approved: 'Human Approved',
      human_rejected: 'Human Rejected',
    };
    return labelMap[statusLower] || status;
  };

  if (loading) {
    return (
      <Container>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error || !content) {
    return (
      <Container>
        <Alert severity="error" sx={{ mt: 3 }}>
          {error || 'Content not found'}
        </Alert>
        <Button startIcon={<ArrowBack />} onClick={() => navigate('/dashboard')} sx={{ mt: 2 }}>
          Back to Dashboard
        </Button>
      </Container>
    );
  }

  // Determine if content needs action
  const needsAction = content.status === 'pending' || content.status === 'flagged' ||
                      content.status === 'pending_human_review' || content.requires_human_review;

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button startIcon={<ArrowBack />} onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Main Content Card */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                <Typography variant="h5" gutterBottom>
                  Content Review
                </Typography>
                <Chip
                  icon={getStatusIcon(content.status)}
                  label={getStatusLabel(content.status)}
                  color={getStatusColor(content.status)}
                />
              </Box>

              <Divider sx={{ mb: 3 }} />

              <Typography variant="h6" gutterBottom>
                Content Text
              </Typography>
              <Paper
                sx={{
                  p: 3,
                  bgcolor: 'grey.50',
                  mb: 3,
                  borderLeft: 4,
                  borderColor: getToxicityColor(content.analysis?.toxicity_score || 0) + '.main',
                }}
              >
                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                  {content.content_text}
                </Typography>
              </Paper>

              {/* Metadata */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Person fontSize="small" color="action" />
                    <Typography variant="body2" color="text.secondary">
                      Author: {content.username || content.author_id || content.user_id || 'Unknown'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CalendarToday fontSize="small" color="action" />
                    <Typography variant="body2" color="text.secondary">
                      {new Date(content.created_at).toLocaleString()}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Language fontSize="small" color="action" />
                    <Typography variant="body2" color="text.secondary">
                      Platform: {content.platform}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {/* AI Moderation Action */}
              {content.status === 'pending' && (
                <Box sx={{ mt: 3 }}>
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    onClick={handleModerate}
                    disabled={moderating}
                    startIcon={moderating ? <CircularProgress size={20} /> : null}
                  >
                    {moderating ? 'Analyzing with AI Agents...' : 'Run AI Moderation'}
                  </Button>
                </Box>
              )}

              {/* Manual Moderation Actions */}
              {needsAction && (
                <Box sx={{ mt: 3 }}>
                  <Divider sx={{ mb: 2 }} />
                  <Typography variant="subtitle2" gutterBottom color="text.secondary">
                    Manual Moderation Actions
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                    <Button
                      variant="contained"
                      color="success"
                      startIcon={<ThumbUp />}
                      onClick={() => handleOpenReview('approve')}
                      sx={{ flex: 1 }}
                    >
                      Approve
                    </Button>
                    <Button
                      variant="contained"
                      color="warning"
                      startIcon={<Flag />}
                      onClick={() => handleOpenReview('warn')}
                      sx={{ flex: 1 }}
                    >
                      Warn
                    </Button>
                    <Button
                      variant="contained"
                      color="error"
                      startIcon={<ThumbDown />}
                      onClick={() => handleOpenReview('remove')}
                      sx={{ flex: 1 }}
                    >
                      Remove
                    </Button>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Sidebar */}
        <Grid item xs={12} md={4}>
          {/* Toxicity Score */}
          {content.analysis?.toxicity_score !== undefined && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Toxicity Analysis
                </Typography>
                <Box sx={{ textAlign: 'center', my: 2 }}>
                  <Typography variant="h2" color={getToxicityColor(content.analysis.toxicity_score) + '.main'}>
                    {(content.analysis.toxicity_score * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {getToxicityLabel(content.analysis.toxicity_score)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={content.analysis.toxicity_score * 100}
                  color={getToxicityColor(content.analysis.toxicity_score)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </CardContent>
            </Card>
          )}

          {/* Policy Violations */}
          {content.analysis?.policy_violations && content.analysis.policy_violations.length > 0 && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Policy Violations
                </Typography>
                <List dense>
                  {content.analysis.policy_violations.map((violation, idx) => (
                    <ListItem key={idx}>
                      <Warning fontSize="small" color="warning" sx={{ mr: 1 }} />
                      <ListItemText
                        primary={violation}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          )}

          {/* AI Decision */}
          {content.moderation_result && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  AI Decision
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  {content.moderation_result.action === 'approve' ? (
                    <CheckCircle color="success" />
                  ) : (
                    <Cancel color="error" />
                  )}
                  <Typography variant="h6">
                    {content.moderation_result.action === 'approve' ? 'Approved' : 'Removed'}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {content.moderation_result.reasoning}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="caption" color="text.secondary">
                  Confidence: {(content.moderation_result.confidence * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Agent Decisions */}
          {content.agent_decisions && Object.keys(content.agent_decisions).length > 0 && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Agent Decisions
                </Typography>
                <List dense>
                  {Object.entries(content.agent_decisions).map(([agent, decision]) => (
                    <ListItem key={agent}>
                      <ListItemText
                        primary={agent}
                        secondary={`${decision.action} (${(decision.confidence * 100).toFixed(0)}%)`}
                        primaryTypographyProps={{ variant: 'body2', fontWeight: 'medium' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      {/* Manual Review Dialog */}
      <Dialog open={reviewDialog.open} onClose={handleCloseReview} maxWidth="sm" fullWidth>
        <DialogTitle sx={{
          bgcolor: reviewDialog.action === 'approve' ? 'success.main' :
                   reviewDialog.action === 'warn' ? 'warning.main' : 'error.main',
          color: 'white'
        }}>
          {reviewDialog.action === 'approve' && 'Approve Content'}
          {reviewDialog.action === 'warn' && 'Warn User'}
          {reviewDialog.action === 'remove' && 'Remove Content'}
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Content Preview:</Typography>
          <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
            <Typography variant="body2">
              {content?.content_text?.substring(0, 300)}
              {content?.content_text?.length > 300 && '...'}
            </Typography>
          </Paper>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Moderation Notes (Optional)"
            placeholder={`Reason for ${reviewDialog.action}ing this content...`}
            value={reviewNotes}
            onChange={(e) => setReviewNotes(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseReview}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSubmitReview}
            disabled={reviewLoading}
            color={reviewDialog.action === 'approve' ? 'success' :
                   reviewDialog.action === 'warn' ? 'warning' : 'error'}
          >
            {reviewLoading ? 'Processing...' : `Confirm ${reviewDialog.action}`}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
