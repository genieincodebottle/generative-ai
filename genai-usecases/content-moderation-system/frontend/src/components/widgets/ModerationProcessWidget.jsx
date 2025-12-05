import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Divider,
  Stack,
  Avatar
} from '@mui/material';

const ProcessStep = ({ number, text, color = 'primary' }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
    <Avatar
      sx={{
        bgcolor: `${color}.main`,
        width: 40,
        height: 40,
        fontWeight: 'bold'
      }}
    >
      {number}
    </Avatar>
    <Typography variant="body1">
      {text}
    </Typography>
  </Box>
);

const ModerationProcessWidget = () => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h5" component="h2" fontWeight="bold" gutterBottom>
          How Moderation Works
        </Typography>
        <Divider sx={{ mb: 3 }} />

        <Stack spacing={1.5}>
          <ProcessStep
            number={1}
            text="You submit a story or comment"
            color="primary"
          />
          <ProcessStep
            number={2}
            text="AI analyzes for policy violations"
            color="secondary"
          />
          <ProcessStep
            number={3}
            text="Flagged content goes to human review"
            color="info"
          />
          <ProcessStep
            number={4}
            text="Decision made (approve/warn/remove)"
            color="warning"
          />
          <ProcessStep
            number={5}
            text="You can appeal if you disagree"
            color="success"
          />
        </Stack>
      </CardContent>
    </Card>
  );
};

export default ModerationProcessWidget;
