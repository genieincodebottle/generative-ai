import {
  Cancel,
  CheckCircle,
  Gavel,
  HourglassEmpty,
  Refresh,
  Visibility,
} from '@mui/icons-material';
import {
  Alert,
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
  FormControlLabel,
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
  Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { appealsAPI, contentAPI } from '../../services/api';
import { useModerationStore } from '../../store/moderationStore';

export default function AppealsManagement() {
  const { pendingAppeals, setPendingAppeals } = useModerationStore();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [selectedAppeal, setSelectedAppeal] = useState(null);
  const [reviewDialogOpen, setReviewDialogOpen] = useState(false);
  const [reviewData, setReviewData] = useState({
    outcome: 'upheld',
    reasoning: '',
  });

  useEffect(() => {
    fetchAppeals();
  }, []);

  const fetchAppeals = async () => {
    try {
      setLoading(true);
      const data = await appealsAPI.getAllAppeals();
      setPendingAppeals(data.appeals || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load appeals');
    } finally {
      setLoading(false);
    }
  };

  const handleViewAppeal = async (appeal) => {
    try {
      // Fetch the full appeal details including content
      const [appealData, contentData] = await Promise.all([
        appealsAPI.getAppealById(appeal.appeal_id),
        contentAPI.getContentById(appeal.content_id),
      ]);

      setSelectedAppeal({
        ...appealData,
        content: contentData,
      });
      setReviewDialogOpen(true);
      setReviewData({ outcome: 'upheld', reasoning: '' });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load appeal details');
    }
  };

  const handleSubmitReview = async () => {
    try {
      setLoading(true);
      await appealsAPI.reviewAppeal(selectedAppeal.appeal_id, reviewData);
      setReviewDialogOpen(false);
      setSelectedAppeal(null);
      await fetchAppeals();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to submit review');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    const statusMap = {
      pending: 'warning',
      upheld: 'success',
      overturned: 'error',
      partial: 'info',
    };
    return statusMap[status?.toLowerCase()] || 'default';
  };

  const getStatusIcon = (status) => {
    const statusLower = status?.toLowerCase();
    if (statusLower === 'upheld') return <CheckCircle color="success" />;
    if (statusLower === 'overturned') return <Cancel color="error" />;
    if (statusLower === 'pending') return <HourglassEmpty color="warning" />;
    return <Gavel color="action" />;
  };

  const filterAppealsByTab = () => {
    if (tabValue === 0) return pendingAppeals;
    if (tabValue === 1) return pendingAppeals.filter((a) => a.status === 'pending');
    if (tabValue === 2) return pendingAppeals.filter((a) => a.status === 'upheld');
    if (tabValue === 3) return pendingAppeals.filter((a) => a.status === 'overturned');
    return pendingAppeals;
  };

  const filteredAppeals = filterAppealsByTab();

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Button variant="outlined" startIcon={<Refresh />} onClick={fetchAppeals}>
            Refresh
          </Button>
        </Box>

        {/* Stats Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Total Appeals
                </Typography>
                <Typography variant="h4">{pendingAppeals.length}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Pending Review
                </Typography>
                <Typography variant="h4" color="warning.main">
                  {pendingAppeals.filter((a) => a.status === 'pending').length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Upheld
                </Typography>
                <Typography variant="h4" color="success.main">
                  {pendingAppeals.filter((a) => a.status === 'upheld').length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom variant="body2">
                  Overturned
                </Typography>
                <Typography variant="h4" color="error.main">
                  {pendingAppeals.filter((a) => a.status === 'overturned').length}
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

        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Learning System:</strong> Overturned appeals automatically trigger agent learning.
            The system adjusts thresholds and confidence scores based on appeal outcomes.
          </Typography>
        </Alert>

        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs value={tabValue} onChange={(e, val) => setTabValue(val)}>
            <Tab label={`All (${pendingAppeals.length})`} />
            <Tab
              label={`Pending (${pendingAppeals.filter((a) => a.status === 'pending').length})`}
            />
            <Tab
              label={`Upheld (${pendingAppeals.filter((a) => a.status === 'upheld').length})`}
            />
            <Tab
              label={`Overturned (${
                pendingAppeals.filter((a) => a.status === 'overturned').length
              })`}
            />
          </Tabs>
        </Paper>

        {loading && <LinearProgress sx={{ mb: 2 }} />}

        {/* Appeals Table */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Appeal ID</TableCell>
                <TableCell>Content ID</TableCell>
                <TableCell>Reason</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Submitted</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredAppeals.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    <Typography color="text.secondary" sx={{ py: 3 }}>
                      No appeals found
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                filteredAppeals.map((appeal) => (
                  <TableRow key={appeal.appeal_id} hover>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {appeal.appeal_id?.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {appeal.content_id?.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>
                        {appeal.reason}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(appeal.status)}
                        label={appeal.status}
                        color={getStatusColor(appeal.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption">
                        {new Date(appeal.created_at).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        startIcon={<Visibility />}
                        onClick={() => handleViewAppeal(appeal)}
                      >
                        Review
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>

      {/* Review Appeal Dialog */}
      <Dialog
        open={reviewDialogOpen}
        onClose={() => setReviewDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Review Appeal</DialogTitle>
        <DialogContent>
          {selectedAppeal && (
            <Box sx={{ pt: 1 }}>
              {/* Original Content */}
              <Typography variant="h6" gutterBottom>
                Original Content
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'grey.50', mb: 3 }}>
                <Typography variant="body1" paragraph>
                  {selectedAppeal.content?.content_text}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Typography variant="caption" color="text.secondary">
                  AI Decision: {selectedAppeal.content?.moderation_result?.action} (
                  {(selectedAppeal.content?.moderation_result?.confidence * 100).toFixed(0)}%
                  confidence)
                </Typography>
              </Paper>

              {/* Appeal Reason */}
              <Typography variant="h6" gutterBottom>
                Appeal Reason
              </Typography>
              <Paper sx={{ p: 2, bgcolor: 'info.50', mb: 3 }}>
                <Typography variant="body1">{selectedAppeal.reason}</Typography>
              </Paper>

              {/* Review Form */}
              <Typography variant="h6" gutterBottom>
                Your Review
              </Typography>
              <RadioGroup
                value={reviewData.outcome}
                onChange={(e) => setReviewData({ ...reviewData, outcome: e.target.value })}
              >
                <FormControlLabel
                  value="upheld"
                  control={<Radio />}
                  label="Uphold - Original decision was correct"
                />
                <FormControlLabel
                  value="overturned"
                  control={<Radio />}
                  label="Overturn - Original decision was wrong"
                />
                <FormControlLabel
                  value="partial"
                  control={<Radio />}
                  label="Partial - Some aspects need revision"
                />
              </RadioGroup>

              <TextField
                fullWidth
                multiline
                rows={4}
                label="Reasoning"
                value={reviewData.reasoning}
                onChange={(e) => setReviewData({ ...reviewData, reasoning: e.target.value })}
                margin="normal"
                placeholder="Explain your decision..."
                required
              />

              {reviewData.outcome === 'overturned' && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>Learning Trigger:</strong> Overturning this decision will cause the AI
                    agents to learn from this mistake. Thresholds and confidence scores will be
                    automatically adjusted.
                  </Typography>
                </Alert>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReviewDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSubmitReview}
            variant="contained"
            disabled={!reviewData.reasoning}
          >
            Submit Review
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
