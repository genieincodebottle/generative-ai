import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Divider,
  Stack
} from '@mui/material';
import {
  Gavel as GavelIcon,
  Add as AddIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  HourglassEmpty as HourglassIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import { useAuthStore } from '../../store/authStore';
import { appealsAPI, storiesAPI } from '../../services/api';

const UserAppeals = () => {
  const { user } = useAuthStore();
  const [appeals, setAppeals] = useState([]);
  const [myStories, setMyStories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openDialog, setOpenDialog] = useState(false);
  const [viewDialog, setViewDialog] = useState(false);
  const [selectedStory, setSelectedStory] = useState(null);
  const [selectedAppeal, setSelectedAppeal] = useState(null);
  const [appealReason, setAppealReason] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    loadData();
  }, [user]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('Loading appeals for user:', user);
      console.log('User ID:', user?.user_id);

      // Load user's appeals
      const appealsResponse = await appealsAPI.getUserAppeals(user?.user_id);
      setAppeals(appealsResponse.appeals || []);

      // Load user's stories (to find rejected/removed ones)
      const storiesResponse = await storiesAPI.getUserStories(user?.user_id, 100);
      console.log('Stories API response:', storiesResponse);

      // The API returns { stories: [...], count: n }, not wrapped in .data
      const userStories = storiesResponse.stories || [];

      // Filter stories that can be appealed (removed, flagged, or pending human review)
      const appealableStories = userStories.filter(story => {
        const status = story.moderation_status?.toLowerCase();
        console.log(`Story "${story.title}" has status: ${story.moderation_status}`);
        return ['removed', 'flagged', 'pending_human_review', 'rejected'].includes(status);
      });
      setMyStories(appealableStories);

      console.log('User stories:', userStories);
      console.log('Appealable stories:', appealableStories);

    } catch (err) {
      console.error('Error loading appeals:', err);
      setError('Failed to load appeals data');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDialog = (story) => {
    setSelectedStory(story);
    setAppealReason('');
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedStory(null);
    setAppealReason('');
  };

  const handleViewAppeal = (appeal) => {
    setSelectedAppeal(appeal);
    setViewDialog(true);
  };

  const handleCloseViewDialog = () => {
    setViewDialog(false);
    setSelectedAppeal(null);
  };

  const handleSubmitAppeal = async () => {
    if (!appealReason.trim()) {
      setError('Please provide a reason for your appeal');
      return;
    }

    try {
      setSubmitting(true);
      setError(null);

      const payload = {
        content_id: String(selectedStory.content_id || selectedStory.story_id),
        user_id: String(user?.user_id),
        appeal_reason: String(appealReason)
      };

      console.log('Submitting appeal payload:', payload);
      await appealsAPI.createAppeal(payload);

      setSuccess('Appeal submitted successfully! We will review it within 48 hours.');
      handleCloseDialog();
      loadData(); // Refresh the appeals list

    } catch (err) {
      console.error('Error submitting appeal:', err);
      const errorDetail = err.response?.data?.detail;
      let errorMessage = 'Failed to submit appeal';

      if (typeof errorDetail === 'string') {
        errorMessage = errorDetail;
      } else if (Array.isArray(errorDetail)) {
        errorMessage = errorDetail.map(e => e.msg || JSON.stringify(e)).join(', ');
      } else if (errorDetail) {
        errorMessage = JSON.stringify(errorDetail);
      }

      setError(errorMessage);
    } finally {
      setSubmitting(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'approved':
      case 'overturned':
        return 'success';
      case 'pending':
        return 'warning';
      case 'rejected':
      case 'upheld':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'approved':
      case 'overturned':
        return <CheckCircleIcon />;
      case 'pending':
        return <HourglassIcon />;
      case 'rejected':
      case 'upheld':
        return <CancelIcon />;
      default:
        return <HourglassIcon />;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" fontWeight="bold" gutterBottom>
          <GavelIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          My Appeals
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Submit and track appeals for content moderation decisions
        </Typography>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      {/* Stories that can be appealed */}
      {myStories.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Content Available for Appeal
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              You can appeal moderation decisions on the following content:
            </Typography>
            <Stack spacing={2}>
              {myStories.map((story) => (
                <Paper key={story.story_id} elevation={1} sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {story.title}
                      </Typography>
                      <Box sx={{ mt: 1, mb: 1 }}>
                        <Chip
                          label={story.moderation_status}
                          size="small"
                          color={getStatusColor(story.moderation_status)}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        Submitted: {formatDate(story.created_at)}
                      </Typography>
                    </Box>
                    <Button
                      variant="outlined"
                      startIcon={<AddIcon />}
                      onClick={() => handleOpenDialog(story)}
                      disabled={appeals.some(a => a.content_id === story.story_id && a.status === 'pending')}
                    >
                      {appeals.some(a => a.content_id === story.story_id && a.status === 'pending')
                        ? 'Appeal Pending'
                        : 'Submit Appeal'}
                    </Button>
                  </Box>
                </Paper>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* Appeals History */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          Appeals History
        </Typography>

        {appeals.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <GavelIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              No Appeals Yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              You haven't submitted any appeals
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Content</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                  <TableCell><strong>Submitted</strong></TableCell>
                  <TableCell><strong>Reviewed</strong></TableCell>
                  <TableCell><strong>Actions</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {appeals.map((appeal) => (
                  <TableRow key={appeal.appeal_id} hover>
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {appeal.content_title || `Story #${appeal.content_id}`}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(appeal.status)}
                        label={appeal.status || 'pending'}
                        size="small"
                        color={getStatusColor(appeal.status)}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatDate(appeal.appeal_date)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {appeal.review_date ? formatDate(appeal.review_date) : '-'}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        startIcon={<VisibilityIcon />}
                        onClick={() => handleViewAppeal(appeal)}
                      >
                        View Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>

      {/* Submit Appeal Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          <Typography variant="h6" fontWeight="bold">
            Submit Appeal
          </Typography>
        </DialogTitle>
        <DialogContent dividers>
          {selectedStory && (
            <>
              <Alert severity="info" sx={{ mb: 3 }}>
                You are appealing the moderation decision for: <strong>{selectedStory.title}</strong>
              </Alert>

              <Typography variant="subtitle2" gutterBottom>
                Current Status:
              </Typography>
              <Chip
                label={selectedStory.moderation_status}
                size="small"
                color={getStatusColor(selectedStory.moderation_status)}
                sx={{ mb: 3 }}
              />

              <TextField
                fullWidth
                multiline
                rows={6}
                label="Reason for Appeal"
                placeholder="Please explain why you believe this moderation decision should be reviewed..."
                value={appealReason}
                onChange={(e) => setAppealReason(e.target.value)}
                inputProps={{ maxLength: 2000 }}
                helperText={`${appealReason.length}/2000 characters. Be specific and provide clear reasoning.`}
              />

              <Alert severity="warning" sx={{ mt: 2 }}>
                Appeals are typically reviewed within 48 hours. Please be patient and avoid submitting duplicate appeals.
              </Alert>
            </>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={handleCloseDialog} disabled={submitting}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSubmitAppeal}
            disabled={submitting || !appealReason.trim()}
            startIcon={submitting ? <CircularProgress size={20} /> : <GavelIcon />}
          >
            {submitting ? 'Submitting...' : 'Submit Appeal'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* View Appeal Details Dialog */}
      <Dialog open={viewDialog} onClose={handleCloseViewDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          <Typography variant="h6" fontWeight="bold">
            Appeal Details
          </Typography>
        </DialogTitle>
        <DialogContent dividers>
          {selectedAppeal && (
            <Box>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Status
              </Typography>
              <Chip
                icon={getStatusIcon(selectedAppeal.status)}
                label={selectedAppeal.status || 'pending'}
                color={getStatusColor(selectedAppeal.status)}
                sx={{ mb: 3 }}
              />

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Your Appeal Reason
              </Typography>
              <Paper variant="outlined" sx={{ p: 2, mb: 3, bgcolor: 'grey.50' }}>
                <Typography variant="body2">
                  {selectedAppeal.reason}
                </Typography>
              </Paper>

              {selectedAppeal.review_notes && (
                <>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Review Decision
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="body2">
                      {selectedAppeal.review_notes}
                    </Typography>
                  </Paper>
                </>
              )}

              <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Submitted
                  </Typography>
                  <Typography variant="body2">
                    {formatDate(selectedAppeal.appeal_date)}
                  </Typography>
                </Box>
                {selectedAppeal.review_date && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Reviewed
                    </Typography>
                    <Typography variant="body2">
                      {formatDate(selectedAppeal.review_date)}
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseViewDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default UserAppeals;
