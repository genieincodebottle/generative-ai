import {
  Psychology,
  Refresh,
  Speed,
  TrendingDown,
  TrendingUp,
} from '@mui/icons-material';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Container,
  FormControl,
  Grid,
  InputLabel,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';
import { useEffect, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { analyticsAPI } from '../../services/api';

export default function Analytics() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [agentPerformance, setAgentPerformance] = useState([]);
  const [learningMetrics, setLearningMetrics] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState('all');
  const [timeRange, setTimeRange] = useState(30);

  useEffect(() => {
    fetchAnalytics();
  }, [selectedAgent, timeRange]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);

      const [metricsData, performanceData, learningData] = await Promise.all([
        analyticsAPI.getSystemMetrics(),
        analyticsAPI.getAgentPerformance(selectedAgent === 'all' ? null : selectedAgent),
        analyticsAPI.getLearningMetrics(selectedAgent === 'all' ? null : selectedAgent, timeRange),
      ]);

      setMetrics(metricsData);
      setAgentPerformance(performanceData.agents || []);
      setLearningMetrics(learningData);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  };

  // Get unique agent names from performance data, or use defaults
  const agentNames = agentPerformance.length > 0
    ? agentPerformance.map(a => a.agent_name)
    : [
        'Content Analysis Agent',
        'Toxicity Detection Agent',
        'Policy Violation Agent',
        'User Reputation Agent',
        'Appeal Review Agent',
        'Action Enforcement Agent',
      ];

  // Use learning trend data from API, or show placeholder if no data
  const learningTrendData = learningMetrics?.learning_trend?.length > 0
    ? learningMetrics.learning_trend
    : [{ session: 0, accuracy: 0, appeals: 0 }];

  const getPerformanceColor = (value) => {
    if (value >= 90) return 'success';
    if (value >= 75) return 'warning';
    return 'error';
  };

  const getTrendIcon = (trend) => {
    if (trend === 'improving') return <TrendingUp color="success" />;
    if (trend === 'declining') return <TrendingDown color="error" />;
    return null;
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Agent</InputLabel>
              <Select
                value={selectedAgent}
                label="Agent"
                onChange={(e) => setSelectedAgent(e.target.value)}
              >
                <MenuItem value="all">All Agents</MenuItem>
                {agentNames.map((name) => (
                  <MenuItem key={name} value={name}>
                    {name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                label="Time Range"
                onChange={(e) => setTimeRange(e.target.value)}
              >
                <MenuItem value={7}>7 Days</MenuItem>
                <MenuItem value={30}>30 Days</MenuItem>
                <MenuItem value={90}>90 Days</MenuItem>
              </Select>
            </FormControl>
            <Button variant="outlined" startIcon={<Refresh />} onClick={fetchAnalytics}>
              Refresh
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Learning System:</strong> Agents continuously improve through episodic and
            semantic memory. Success rates increase as the system learns from appeal outcomes.
          </Typography>
        </Alert>

        {loading && <LinearProgress sx={{ mb: 2 }} />}

        {/* Key Metrics */}
        {metrics && (
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Speed color="primary" />
                    <Typography color="text.secondary" variant="body2">
                      Overall Accuracy
                    </Typography>
                  </Box>
                  <Typography variant="h4" color={metrics.overall_accuracy >= 80 ? "success.main" : "warning.main"}>
                    {metrics.overall_accuracy ?? 0}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Based on {metrics.total_decisions ?? 0} decisions
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Psychology color="secondary" />
                    <Typography color="text.secondary" variant="body2">
                      Total Decisions
                    </Typography>
                  </Box>
                  <Typography variant="h4">{metrics.total_decisions ?? 0}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Across all agents
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <TrendingDown color={metrics.appeal_rate <= 10 ? "success" : "warning"} />
                    <Typography color="text.secondary" variant="body2">
                      Appeal Rate
                    </Typography>
                  </Box>
                  <Typography variant="h4" color={metrics.appeal_rate <= 10 ? "success.main" : "warning.main"}>
                    {metrics.appeal_rate ?? 0}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {metrics.overturned_appeals ?? 0} overturned
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <TrendingUp color="info" />
                    <Typography color="text.secondary" variant="body2">
                      Learning Sessions
                    </Typography>
                  </Box>
                  <Typography variant="h4" color="info.main">
                    {metrics.learning_sessions ?? 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Improvement cycles
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Learning Progress Chart */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Learning Progress Over Time
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Accuracy increases as agents learn from overturned appeals
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={learningTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="session" label={{ value: 'Sessions', position: 'insideBottom', offset: -5 }} />
                <YAxis yAxisId="left" label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'Appeal Rate (%)', angle: 90, position: 'insideRight' }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#2e7d32" strokeWidth={2} name="Accuracy %" />
                <Line yAxisId="right" type="monotone" dataKey="appeals" stroke="#d32f2f" strokeWidth={2} name="Appeal Rate %" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Agent Performance */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Agent Performance Breakdown
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Agent</TableCell>
                    <TableCell>Accuracy</TableCell>
                    <TableCell>Decisions</TableCell>
                    <TableCell>Avg Confidence</TableCell>
                    <TableCell>Trend</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {agentPerformance.length > 0 ? (
                    agentPerformance.map((agent) => (
                      <TableRow key={agent.agent_name}>
                        <TableCell>{agent.agent_name}</TableCell>
                        <TableCell>
                          <Chip
                            label={`${agent.accuracy ?? 0}%`}
                            color={getPerformanceColor(agent.accuracy ?? 0)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>{agent.total_decisions ?? 0}</TableCell>
                        <TableCell>{agent.average_confidence ?? 0}%</TableCell>
                        <TableCell>{getTrendIcon(agent.trend)}</TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={5} align="center">
                        <Typography color="text.secondary">
                          No agent performance data available yet
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>

        {/* Memory System Stats */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Episodic Memory
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Specific moderation cases stored for similarity matching
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="Total Episodes"
                      secondary={learningMetrics?.episodic_count ?? 0}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Avg Retrieval Time"
                      secondary={`${learningMetrics?.avg_retrieval_time ?? 0}ms`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Most Similar Cases Used"
                      secondary={learningMetrics?.similar_cases_used ?? 0}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Semantic Memory
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Learned patterns and adjusted thresholds
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="Learned Patterns"
                      secondary={learningMetrics?.learned_patterns ?? 0}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Threshold Adjustments"
                      secondary={learningMetrics?.threshold_adjustments ?? 0}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Confidence Calibrations"
                      secondary={learningMetrics?.confidence_calibrations ?? 0}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}
