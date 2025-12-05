import {
  Block,
  Cancel,
  CheckCircle,
  Flag,
  HourglassEmpty,
  PersonOff,
  Refresh,
  ReportProblem,
  Search,
  ThumbDown,
  ThumbUp,
  Visibility,
  Warning,
  VerifiedUser,
  TrendingUp,
} from '@mui/icons-material';
import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  IconButton,
  InputAdornment,
  LinearProgress,
  Paper,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api, { analyticsAPI, contentAPI, storiesAPI } from '../../services/api';
import { useAuthStore } from '../../store/authStore';
import { useModerationStore } from '../../store/moderationStore';

export default function Dashboard() {
  const navigate = useNavigate();
  const { pendingContent, setPendingContent, setContentLoading } = useModerationStore();
  const { isSeniorModerator } = useAuthStore();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Quick action dialog
  const [actionDialog, setActionDialog] = useState({ open: false, content: null, action: '' });
  const [actionNotes, setActionNotes] = useState('');
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      setContentLoading(true);

      // Fetch ALL content and metrics in parallel
      // Note: We only fetch from content_submissions to avoid duplicates
      // since stories are created in both content_submissions AND stories tables
      const [contentData, metricsData] = await Promise.all([
        contentAPI.getAllContent(null, 100),
        analyticsAPI.getSystemMetrics().catch(() => null),
      ]);

      // Use content submissions only (includes all story submissions)
      const allContent = contentData.content || [];

      setPendingContent(allContent);
      setMetrics(metricsData);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
      setContentLoading(false);
    }
  };

  const handleViewContent = (contentId) => {
    navigate(`/review/${contentId}`);
  };

  const handleQuickAction = (content, action) => {
    setActionDialog({ open: true, content, action });
    setActionNotes('');
  };

  const handleSubmitAction = async () => {
    if (!actionDialog.content || !actionDialog.action) return;

    setActionLoading(true);
    try {
      await api.post(`/api/content/${actionDialog.content.content_id}/manual-review`, {
        decision: actionDialog.action,
        reviewer_name: 'Dashboard Quick Action',
        notes: actionNotes || `Quick ${actionDialog.action} from dashboard`,
      });

      setSuccess(`Content ${actionDialog.action}d successfully`);
      setActionDialog({ open: false, content: null, action: '' });
      await fetchData();

      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err.response?.data?.detail || `Failed to ${actionDialog.action} content`);
    } finally {
      setActionLoading(false);
    }
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
    if (statusLower === 'approved' || statusLower === 'human_approved') return <CheckCircle color="success" />;
    if (statusLower === 'removed' || statusLower === 'human_rejected') return <Cancel color="error" />;
    if (statusLower === 'flagged' || statusLower === 'pending_human_review') return <Warning color="warning" />;
    if (statusLower === 'escalated') return <TrendingUp color="error" />;
    if (statusLower === 'human_review_completed') return <VerifiedUser color="info" />;
    if (statusLower === 'warned') return <Flag color="warning" />;
    return <HourglassEmpty color="action" />;
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

  const filterContentByTab = () => {
    if (tabValue === 0) return pendingContent;
    if (tabValue === 1) return pendingContent.filter((c) =>
      (c.status === 'pending' || c.current_status === 'pending') && !c.requires_human_review
    );
    if (tabValue === 2) return pendingContent.filter((c) =>
      c.status === 'flagged' || c.current_status === 'flagged' ||
      c.status === 'pending_human_review' || c.requires_human_review ||
      c.status === 'escalated' || c.current_status === 'escalated'
    );
    if (tabValue === 3) return pendingContent.filter((c) =>
      c.status === 'approved' || c.current_status === 'approved' ||
      c.status === 'human_approved' || c.current_status === 'human_approved'
    );
    if (tabValue === 4) return pendingContent.filter((c) =>
      c.status === 'removed' || c.current_status === 'removed' ||
      c.status === 'human_rejected' || c.current_status === 'human_rejected'
    );
    return pendingContent;
  };

  const applySearchFilter = (content) => {
    if (!searchQuery.trim()) return content;

    const query = searchQuery.toLowerCase().trim();
    return content.filter((item) => {
      // Search in content text
      const contentText = (item.content_text || '').toLowerCase();

      // Search in content ID
      const contentId = (item.content_id || '').toLowerCase();

      // Search in author name/username
      const author = (item.username || item.author_id || item.user_id || '').toString().toLowerCase();

      // Search in platform
      const platform = (item.platform || '').toLowerCase();

      return contentText.includes(query) ||
             contentId.includes(query) ||
             author.includes(query) ||
             platform.includes(query);
    });
  };

  const filteredContent = applySearchFilter(filterContentByTab());

  const needsReviewCount = pendingContent.filter(c =>
    c.status === 'flagged' || c.status === 'pending_human_review' || c.requires_human_review ||
    c.status === 'pending' || c.current_status === 'pending' ||
    c.status === 'escalated' || c.current_status === 'escalated'
  ).length;

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, gap: 2 }}>
          <TextField
            placeholder="Search by content, author, or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            size="small"
            sx={{ flexGrow: 1, maxWidth: 500 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
          />
          <Button variant="outlined" startIcon={<Refresh />} onClick={fetchData}>
            Refresh
          </Button>
        </Box>

        {/* Metrics Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Total Content
                </Typography>
                <Typography variant="h4">{metrics?.total_content || pendingContent.length}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ borderLeft: 4, borderColor: 'warning.main' }}>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Needs Review
                </Typography>
                <Typography variant="h4" color="warning.main">
                  {needsReviewCount}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ borderLeft: 4, borderColor: 'success.main' }}>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Approved
                </Typography>
                <Typography variant="h4" color="success.main">
                  {metrics?.approved_content || pendingContent.filter(c => c.status === 'approved').length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ borderLeft: 4, borderColor: 'error.main' }}>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Removed
                </Typography>
                <Typography variant="h4" color="error.main">
                  {metrics?.removed_content || pendingContent.filter(c => c.status === 'removed').length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

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

        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs value={tabValue} onChange={(e, val) => setTabValue(val)}>
            <Tab label={`All (${pendingContent.length})`} />
            <Tab label={`Pending (${pendingContent.filter((c) => c.status === 'pending' && !c.requires_human_review).length})`} />
            <Tab
              label={
                <Badge badgeContent={pendingContent.filter(c =>
                  c.status === 'flagged' || c.requires_human_review ||
                  c.status === 'pending_human_review' || c.status === 'escalated'
                ).length} color="error">
                  <Box sx={{ pr: 2 }}>Flagged / HITL</Box>
                </Badge>
              }
            />
            <Tab label={`Approved (${pendingContent.filter((c) => c.status === 'approved').length})`} />
            <Tab label={`Removed (${pendingContent.filter((c) => c.status === 'removed').length})`} />
          </Tabs>
        </Paper>

        {loading && <LinearProgress sx={{ mb: 2 }} />}

        {/* Content Table */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Content</TableCell>
                <TableCell>Author</TableCell>
                <TableCell>Platform</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Toxicity</TableCell>
                <TableCell>Created</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredContent.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={8} align="center">
                    <Typography color="text.secondary" sx={{ py: 3 }}>
                      No content found
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                filteredContent.map((content) => {
                  const status = content.status || content.current_status;
                  const needsAction = status === 'pending' || status === 'flagged' ||
                                     status === 'pending_human_review' || content.requires_human_review ||
                                     status === 'escalated';
                  const toxicity = content.toxicity_score || content.analysis?.toxicity_score || 0;

                  return (
                    <TableRow
                      key={content.content_id}
                      hover
                      sx={{
                        bgcolor: needsAction && (status === 'flagged' || content.requires_human_review) ? 'warning.50' :
                                status === 'escalated' ? 'error.50' : 'inherit',
                      }}
                    >
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {content.content_id?.substring(0, 8)}...
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 250 }}>
                          {content.content_text}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap>
                          {content.username || content.author_id || content.user_id || 'Unknown'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={content.platform} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getStatusIcon(status)}
                          label={getStatusLabel(status)}
                          color={getStatusColor(status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {toxicity > 0 ? (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={toxicity * 100}
                              color={toxicity > 0.7 ? 'error' : toxicity > 0.4 ? 'warning' : 'success'}
                              sx={{ width: 50, height: 6, borderRadius: 1 }}
                            />
                            <Typography variant="caption">
                              {(toxicity * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        ) : (
                          <Typography variant="caption" color="text.secondary">-</Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {new Date(content.created_at || content.submission_timestamp).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'center', flexWrap: 'wrap' }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleViewContent(content.content_id)}
                            >
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          {needsAction && (
                            <>
                              <Tooltip title="Approve">
                                <IconButton
                                  size="small"
                                  color="success"
                                  onClick={() => handleQuickAction(content, 'approve')}
                                >
                                  <ThumbUp />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Warn User">
                                <IconButton
                                  size="small"
                                  color="warning"
                                  onClick={() => handleQuickAction(content, 'warn')}
                                >
                                  <Flag />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Remove">
                                <IconButton
                                  size="small"
                                  color="error"
                                  onClick={() => handleQuickAction(content, 'remove')}
                                >
                                  <ThumbDown />
                                </IconButton>
                              </Tooltip>
                              {isSeniorModerator() && (
                                <>
                                  <Tooltip title="Suspend User">
                                    <IconButton
                                      size="small"
                                      color="error"
                                      onClick={() => handleQuickAction(content, 'suspend')}
                                    >
                                      <Block />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="Ban User">
                                    <IconButton
                                      size="small"
                                      color="error"
                                      onClick={() => handleQuickAction(content, 'ban')}
                                    >
                                      <PersonOff />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="Escalate">
                                    <IconButton
                                      size="small"
                                      color="info"
                                      onClick={() => handleQuickAction(content, 'escalate')}
                                    >
                                      <ReportProblem />
                                    </IconButton>
                                  </Tooltip>
                                </>
                              )}
                            </>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>

      {/* Quick Action Dialog */}
      <Dialog open={actionDialog.open} onClose={() => setActionDialog({ open: false, content: null, action: '' })} maxWidth="sm" fullWidth>
        <DialogTitle sx={{
          bgcolor: actionDialog.action === 'approve' ? 'success.main' :
                   actionDialog.action === 'warn' ? 'warning.main' :
                   actionDialog.action === 'escalate' ? 'info.main' : 'error.main',
          color: 'white'
        }}>
          {actionDialog.action === 'approve' && 'Approve Content'}
          {actionDialog.action === 'warn' && 'Warn User'}
          {actionDialog.action === 'remove' && 'Remove Content'}
          {actionDialog.action === 'suspend' && 'Suspend User'}
          {actionDialog.action === 'ban' && 'Ban User'}
          {actionDialog.action === 'escalate' && 'Escalate to Admin'}
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          {actionDialog.content && (
            <>
              <Typography variant="subtitle2" gutterBottom>Content Preview:</Typography>
              <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                <Typography variant="body2">
                  {actionDialog.content.content_text?.substring(0, 300)}
                  {actionDialog.content.content_text?.length > 300 && '...'}
                </Typography>
              </Paper>
              {(actionDialog.action === 'suspend' || actionDialog.action === 'ban') && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  This action will affect the user's account. Please ensure this is appropriate.
                </Alert>
              )}
              <TextField
                fullWidth
                multiline
                rows={3}
                label={actionDialog.action === 'escalate' ? 'Escalation Reason (Required)' : 'Moderation Notes (Optional)'}
                placeholder={
                  actionDialog.action === 'suspend' ? 'Reason for suspending this user...' :
                  actionDialog.action === 'ban' ? 'Reason for banning this user...' :
                  actionDialog.action === 'escalate' ? 'Why does this need admin attention...' :
                  `Reason for ${actionDialog.action}ing this content...`
                }
                value={actionNotes}
                onChange={(e) => setActionNotes(e.target.value)}
                required={actionDialog.action === 'escalate'}
              />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActionDialog({ open: false, content: null, action: '' })}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSubmitAction}
            disabled={actionLoading || (actionDialog.action === 'escalate' && !actionNotes.trim())}
            color={actionDialog.action === 'approve' ? 'success' :
                   actionDialog.action === 'warn' ? 'warning' :
                   actionDialog.action === 'escalate' ? 'info' : 'error'}
          >
            {actionLoading ? 'Processing...' : `Confirm ${actionDialog.action}`}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
