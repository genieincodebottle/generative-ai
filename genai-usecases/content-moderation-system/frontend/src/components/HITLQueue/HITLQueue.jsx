import {
  Block,
  Cancel,
  CheckCircle,
  Flag,
  Gavel,
  Person,
  Psychology,
  Refresh,
  Report,
  ThumbDown,
  ThumbUp,
  Timer,
  Warning
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
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  Grid,
  LinearProgress,
  Paper,
  Radio,
  RadioGroup,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Typography
} from '@mui/material';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import { useAuthStore } from '../../store/authStore';

const priorityConfig = {
  critical: { color: 'error', label: 'CRITICAL', weight: 100 },
  high: { color: 'warning', label: 'HIGH', weight: 75 },
  medium: { color: 'info', label: 'MEDIUM', weight: 50 },
  low: { color: 'default', label: 'LOW', weight: 25 },
};

const decisionOptions = [
  { value: 'approve', label: 'Approve Content', icon: <ThumbUp />, color: 'success', description: 'Allow content to remain visible' },
  { value: 'warn', label: 'Warn User', icon: <Warning />, color: 'warning', description: 'Issue a warning to the user' },
  { value: 'remove', label: 'Remove Content', icon: <ThumbDown />, color: 'error', description: 'Remove content from platform' },
  { value: 'suspend_user', label: 'Suspend User', icon: <Block />, color: 'error', description: 'Temporarily suspend user account' },
  { value: 'ban_user', label: 'Ban User', icon: <Cancel />, color: 'error', description: 'Permanently ban user account' },
  { value: 'escalate', label: 'Escalate', icon: <Report />, color: 'info', description: 'Escalate to senior moderator' },
];

export default function HITLQueue() {
  const navigate = useNavigate();
  const { user, canReviewHITL } = useAuthStore();
  const [loading, setLoading] = useState(true);
  const [queue, setQueue] = useState([]);
  const [stats, setStats] = useState({ total: 0, critical: 0, high: 0, medium: 0, low: 0 });
  const [tabValue, setTabValue] = useState(0);
  const [error, setError] = useState(null);

  // Review dialog state
  const [reviewDialog, setReviewDialog] = useState({ open: false, item: null });
  const [reviewDecision, setReviewDecision] = useState('');
  const [reviewNotes, setReviewNotes] = useState('');
  const [submittingReview, setSubmittingReview] = useState(false);

  useEffect(() => {
    loadQueue();
  }, []);

  const loadQueue = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/api/hitl/queue');
      setQueue(response.data.queue || []);

      // Calculate stats
      const items = response.data.queue || [];
      setStats({
        total: items.length,
        critical: items.filter(i => i.priority === 'critical').length,
        high: items.filter(i => i.priority === 'high').length,
        medium: items.filter(i => i.priority === 'medium').length,
        low: items.filter(i => i.priority === 'low').length,
      });
    } catch (err) {
      console.error('Error loading HITL queue:', err);
      setError('Failed to load review queue');
    } finally {
      setLoading(false);
    }
  };

  const getFilteredQueue = () => {
    const priorities = ['all', 'critical', 'high', 'medium', 'low'];
    const filterPriority = priorities[tabValue];

    if (filterPriority === 'all') return queue;
    return queue.filter(item => item.priority === filterPriority);
  };

  const handleOpenReview = async (item) => {
    try {
      // Fetch detailed review info
      const response = await api.get(`/api/hitl/review/${item.content_id}`);
      setReviewDialog({ open: true, item: response.data });
      setReviewDecision(item.ai_recommendation || '');
      setReviewNotes('');
    } catch (err) {
      console.error('Error fetching review details:', err);
      // Open with basic info if detailed fetch fails
      setReviewDialog({ open: true, item });
      setReviewDecision(item.ai_recommendation || '');
      setReviewNotes('');
    }
  };

  const handleSubmitReview = async () => {
    if (!reviewDecision || !reviewNotes.trim()) return;

    setSubmittingReview(true);
    try {
      await api.post(`/api/hitl/review/${reviewDialog.item.content_id}`, {
        decision: reviewDecision,
        reviewer_name: user?.full_name || user?.username,
        notes: reviewNotes,
      });

      setReviewDialog({ open: false, item: null });
      loadQueue();
    } catch (err) {
      console.error('Error submitting review:', err);
    } finally {
      setSubmittingReview(false);
    }
  };

  const formatWaitTime = (timestamp) => {
    if (!timestamp) return 'Unknown';
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ${minutes % 60}m`;
    return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  };

  const getPriorityChip = (priority) => {
    const config = priorityConfig[priority] || priorityConfig.low;
    return <Chip label={config.label} color={config.color} size="small" />;
  };

  if (!canReviewHITL()) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          You don't have permission to access the Manual Review queue. This feature is available to Senior Moderators and above.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="body1" color="text.secondary">
          Content flagged by AI for human review. Priority is based on severity, confidence, and user impact.
        </Typography>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6} sm={4} md={2}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4">{stats.total}</Typography>
              <Typography variant="body2" color="text.secondary">Total Queue</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={{ borderLeft: 4, borderColor: 'error.main' }}>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="error.main">{stats.critical}</Typography>
              <Typography variant="body2" color="text.secondary">Critical</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={{ borderLeft: 4, borderColor: 'warning.main' }}>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="warning.main">{stats.high}</Typography>
              <Typography variant="body2" color="text.secondary">High</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={{ borderLeft: 4, borderColor: 'info.main' }}>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="info.main">{stats.medium}</Typography>
              <Typography variant="body2" color="text.secondary">Medium</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={{ borderLeft: 4, borderColor: 'grey.400' }}>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4">{stats.low}</Typography>
              <Typography variant="body2" color="text.secondary">Low</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={loadQueue}
                disabled={loading}
                fullWidth
              >
                Refresh
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Tabs for filtering */}
      <Card>
        <Tabs
          value={tabValue}
          onChange={(e, v) => setTabValue(v)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label={<Badge badgeContent={stats.total} color="primary">All</Badge>} />
          <Tab label={<Badge badgeContent={stats.critical} color="error">Critical</Badge>} />
          <Tab label={<Badge badgeContent={stats.high} color="warning">High</Badge>} />
          <Tab label={<Badge badgeContent={stats.medium} color="info">Medium</Badge>} />
          <Tab label={<Badge badgeContent={stats.low} color="default">Low</Badge>} />
        </Tabs>

        {loading ? (
          <LinearProgress />
        ) : getFilteredQueue().length === 0 ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <CheckCircle sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Queue is empty!
            </Typography>
            <Typography color="text.secondary">
              No content pending human review at this priority level.
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Priority</TableCell>
                  <TableCell>Content Preview</TableCell>
                  <TableCell>Trigger Reasons</TableCell>
                  <TableCell>AI Recommendation</TableCell>
                  <TableCell>Toxicity</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell>Wait Time</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {getFilteredQueue().map((item, index) => (
                  <TableRow
                    key={item.content_id}
                    sx={{
                      bgcolor: item.priority === 'critical' ? 'error.50' :
                               item.priority === 'high' ? 'warning.50' : 'inherit'
                    }}
                  >
                    <TableCell>{item.queue_position || index + 1}</TableCell>
                    <TableCell>{getPriorityChip(item.priority)}</TableCell>
                    <TableCell sx={{ maxWidth: 250 }}>
                      <Typography variant="body2" noWrap>
                        {item.content_preview}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {(item.trigger_reasons || []).slice(0, 2).map((reason, i) => (
                          <Chip
                            key={i}
                            label={reason.replace(/_/g, ' ')}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                        {(item.trigger_reasons || []).length > 2 && (
                          <Chip label={`+${item.trigger_reasons.length - 2}`} size="small" />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={item.ai_recommendation || 'N/A'}
                        size="small"
                        color={
                          item.ai_recommendation === 'approve' ? 'success' :
                          item.ai_recommendation === 'remove' ? 'error' :
                          item.ai_recommendation === 'warn' ? 'warning' : 'default'
                        }
                      />
                      <Typography variant="caption" display="block" color="text.secondary">
                        {((item.ai_confidence || 0) * 100).toFixed(0)}% conf
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={(item.toxicity_score || 0) * 100}
                          color={
                            item.toxicity_score > 0.7 ? 'error' :
                            item.toxicity_score > 0.4 ? 'warning' : 'success'
                          }
                          sx={{ width: 60, height: 8, borderRadius: 1 }}
                        />
                        <Typography variant="caption">
                          {((item.toxicity_score || 0) * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Person fontSize="small" />
                        <Box>
                          <Typography variant="body2">
                            {item.user_info?.username || 'Unknown'}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Rep: {((item.user_info?.reputation || 0) * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Timer fontSize="small" color="action" />
                        <Typography variant="body2">
                          {formatWaitTime(item.waiting_since)}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="contained"
                        size="small"
                        onClick={() => handleOpenReview(item)}
                      >
                        Review
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Card>

      {/* Review Dialog */}
      <Dialog
        open={reviewDialog.open}
        onClose={() => setReviewDialog({ open: false, item: null })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Gavel /> Human Review Required
          {reviewDialog.item?.hitl_context?.priority && (
            <Chip
              label={reviewDialog.item.hitl_context.priority.toUpperCase()}
              color={priorityConfig[reviewDialog.item.hitl_context.priority]?.color || 'default'}
              size="small"
            />
          )}
        </DialogTitle>
        <DialogContent>
          {reviewDialog.item && (
            <Grid container spacing={2}>
              {/* Content Section */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Content to Review:
                </Typography>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    bgcolor: 'grey.50',
                    borderLeft: 4,
                    borderColor: reviewDialog.item.ai_analysis?.toxicity_score > 0.5 ? 'error.main' : 'grey.300'
                  }}
                >
                  <Typography variant="body1">
                    {reviewDialog.item.content?.text || reviewDialog.item.content_preview}
                  </Typography>
                </Paper>
              </Grid>

              {/* AI Analysis */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Psychology /> AI Analysis
                    </Typography>
                    <Divider sx={{ mb: 1 }} />
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Toxicity Score:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {((reviewDialog.item.ai_analysis?.toxicity_score || reviewDialog.item.toxicity_score || 0) * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">AI Recommendation:</Typography>
                        <Chip
                          label={reviewDialog.item.react_analysis?.decision || reviewDialog.item.ai_recommendation || 'N/A'}
                          size="small"
                          color={
                            (reviewDialog.item.react_analysis?.decision || reviewDialog.item.ai_recommendation) === 'approve' ? 'success' :
                            (reviewDialog.item.react_analysis?.decision || reviewDialog.item.ai_recommendation) === 'remove' ? 'error' : 'warning'
                          }
                        />
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Confidence:</Typography>
                        <Typography variant="body2">
                          {((reviewDialog.item.react_analysis?.confidence || reviewDialog.item.ai_confidence || 0) * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>
                    {(reviewDialog.item.ai_analysis?.policy_violations || reviewDialog.item.violations || []).length > 0 && (
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" color="text.secondary">Violations:</Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                          {(reviewDialog.item.ai_analysis?.policy_violations || reviewDialog.item.violations || []).map((v, i) => (
                            <Chip key={i} label={v} size="small" color="error" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* User Info */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Person /> User Profile
                    </Typography>
                    <Divider sx={{ mb: 1 }} />
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Username:</Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {reviewDialog.item.user_profile?.username || reviewDialog.item.user_info?.username || 'Unknown'}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Reputation:</Typography>
                        <Typography variant="body2">
                          {((reviewDialog.item.user_profile?.reputation_score || reviewDialog.item.user_info?.reputation || 0) * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Account Age:</Typography>
                        <Typography variant="body2">
                          {reviewDialog.item.user_profile?.account_age_days || reviewDialog.item.user_info?.account_age_days || 0} days
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">Previous Violations:</Typography>
                        <Typography variant="body2" color={
                          (reviewDialog.item.user_profile?.total_violations || reviewDialog.item.user_info?.total_violations || 0) > 0 ? 'error.main' : 'success.main'
                        }>
                          {reviewDialog.item.user_profile?.total_violations || reviewDialog.item.user_info?.total_violations || 0}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Trigger Reasons */}
              <Grid item xs={12}>
                <Alert severity="info" icon={<Flag />}>
                  <Typography variant="subtitle2">Why Human Review is Required:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                    {(reviewDialog.item.hitl_context?.trigger_reasons || reviewDialog.item.trigger_reasons || []).map((reason, i) => (
                      <Chip key={i} label={reason.replace(/_/g, ' ')} size="small" />
                    ))}
                  </Box>
                </Alert>
              </Grid>

              {/* Decision Section */}
              <Grid item xs={12}>
                <Divider sx={{ my: 1 }} />
                <FormControl component="fieldset" fullWidth>
                  <FormLabel component="legend">Your Decision</FormLabel>
                  <RadioGroup
                    value={reviewDecision}
                    onChange={(e) => setReviewDecision(e.target.value)}
                  >
                    <Grid container spacing={1} sx={{ mt: 1 }}>
                      {decisionOptions.map((option) => (
                        <Grid item xs={12} sm={6} md={4} key={option.value}>
                          <Paper
                            variant={reviewDecision === option.value ? 'elevation' : 'outlined'}
                            elevation={reviewDecision === option.value ? 3 : 0}
                            sx={{
                              p: 1.5,
                              cursor: 'pointer',
                              border: reviewDecision === option.value ? 2 : 1,
                              borderColor: reviewDecision === option.value ? `${option.color}.main` : 'divider',
                              bgcolor: reviewDecision === option.value ? `${option.color}.50` : 'background.paper',
                            }}
                            onClick={() => setReviewDecision(option.value)}
                          >
                            <FormControlLabel
                              value={option.value}
                              control={<Radio size="small" />}
                              label={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  {option.icon}
                                  <Box>
                                    <Typography variant="body2" fontWeight="bold">
                                      {option.label}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                      {option.description}
                                    </Typography>
                                  </Box>
                                </Box>
                              }
                              sx={{ m: 0, width: '100%' }}
                            />
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </RadioGroup>
                </FormControl>
              </Grid>

              {/* Notes */}
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  label="Review Notes (Required)"
                  placeholder="Explain your decision and any relevant context..."
                  value={reviewNotes}
                  onChange={(e) => setReviewNotes(e.target.value)}
                  required
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReviewDialog({ open: false, item: null })}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSubmitReview}
            disabled={!reviewDecision || !reviewNotes.trim() || submittingReview}
            color={
              reviewDecision === 'approve' ? 'success' :
              reviewDecision === 'remove' || reviewDecision === 'ban_user' ? 'error' : 'primary'
            }
          >
            {submittingReview ? 'Submitting...' : 'Submit Decision'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
