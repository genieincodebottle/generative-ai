# Content Moderation Platform - Frontend

A modern React-based frontend for the AI-powered content moderation system with learning capabilities.

## Tech Stack

- **React 18** - Modern UI library
- **Vite** - Fast build tool and dev server
- **Material-UI (MUI)** - Component library
- **React Router** - Client-side routing
- **Zustand** - State management
- **Axios** - HTTP client
- **Recharts** - Data visualization

## Features

### 🎯 Core Functionality
- Real-time content moderation dashboard
- AI agent decision visualization
- Appeal management with learning triggers
- Comprehensive analytics and metrics
- Multi-agent performance tracking

### 🧠 Learning System Visualization
- Episodic memory tracking
- Semantic pattern learning
- Threshold adjustment monitoring
- Success rate improvement charts
- Appeal rate trending

### 📊 Analytics Dashboard
- Overall system accuracy
- Agent-specific performance
- Learning progress over time
- Memory system statistics
- Decision confidence tracking

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Auth/
│   │   │   └── Login.jsx              # Authentication
│   │   ├── Layout/
│   │   │   └── Layout.jsx             # App layout with navigation
│   │   ├── Dashboard/
│   │   │   └── Dashboard.jsx          # Content queue & overview
│   │   ├── ContentReview/
│   │   │   └── ContentReview.jsx      # Individual content review
│   │   ├── Appeals/
│   │   │   └── AppealsManagement.jsx  # Appeal review & learning
│   │   └── Analytics/
│   │       └── Analytics.jsx          # Learning metrics
│   ├── services/
│   │   └── api.js                     # API client & endpoints
│   ├── store/
│   │   ├── authStore.js               # Auth state
│   │   └── moderationStore.js         # Moderation state
│   ├── App.jsx                        # Main app & routing
│   └── main.jsx                       # Entry point
├── index.html
├── vite.config.js
└── package.json
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend server running on `http://localhost:8002`

### Installation

1. **Install dependencies:**
```bash
cd content-moderation-system/frontend
npm install
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env if backend is on different URL
```

3. **Start development server:**
```bash
npm run dev
```

The app will open at `http://localhost:3000`

### Production Build

```bash
npm run build
npm run preview  # Preview production build
```

## Usage Guide

### 1. Login

**Demo Accounts:**
- **Moderator:** `moderator` / `moderator123`
- **Admin:** `admin` / `admin123`

Or use the quick login buttons on the login page.

### 2. Dashboard

**Features:**
- View all content pending moderation
- Filter by status (pending, flagged, approved, removed)
- Submit test content for moderation
- View real-time metrics
- Navigate to individual content reviews

**Actions:**
- Click "Submit Test Content" to add content
- Click eye icon to review content in detail
- Use tabs to filter by status

### 3. Content Review

**Features:**
- View full content details
- See AI agent analysis (toxicity, policy violations)
- View individual agent decisions
- Trigger AI moderation

**Workflow:**
1. Navigate from dashboard
2. Review content and AI analysis
3. Click "Run AI Moderation" if pending
4. View toxicity score and policy violations
5. See final AI decision with reasoning

### 4. Appeals Management

**Features:**
- Review user appeals against moderation decisions
- Uphold or overturn AI decisions
- Trigger automatic agent learning

**Workflow:**
1. View pending appeals
2. Click "Review" to see appeal details
3. Review original content and AI decision
4. Choose outcome (upheld/overturned/partial)
5. Submit reasoning
6. System automatically learns from overturned decisions

**Learning Trigger:**
- When you overturn an appeal, agents automatically:
  - Record the failed decision in episodic memory
  - Adjust confidence scores in semantic memory
  - Update toxicity thresholds
  - Calibrate future decisions

### 5. Analytics

**Features:**
- Overall system accuracy metrics
- Learning progress over time
- Agent performance breakdown
- Memory system statistics
- Trend visualization

**Metrics Explained:**
- **Overall Accuracy:** Success rate of all decisions (increases over time)
- **Appeal Rate:** Percentage of decisions appealed (decreases as agents learn)
- **Episodic Memory:** Number of specific cases stored for reference
- **Semantic Memory:** Learned patterns and threshold adjustments

## API Integration

### Endpoints Used

**Authentication:**
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout

**Content:**
- `GET /api/content/pending` - Get pending content
- `GET /api/content/:id` - Get content details
- `POST /api/content/submit` - Submit new content
- `POST /api/content/:id/moderate` - Trigger AI moderation

**Appeals:**
- `GET /api/appeals/all` - Get all appeals
- `GET /api/appeals/:id` - Get appeal details
- `POST /api/appeals/:id/review` - Review appeal (triggers learning)

**Analytics:**
- `GET /api/analytics/metrics` - System metrics
- `GET /api/analytics/agent-performance` - Agent stats
- `GET /api/analytics/learning` - Learning metrics

### Configuration

Backend URL can be configured in `.env`:
```env
VITE_API_URL=http://localhost:8002
```

Or via Vite proxy in `vite.config.js`:
```js
server: {
  proxy: {
    '/api': 'http://localhost:8002'
  }
}
```

## State Management

### Auth Store (`authStore.js`)

```javascript
const { user, isAuthenticated, login, logout } = useAuthStore();
```

**Methods:**
- `login(user, token)` - Authenticate user
- `logout()` - Clear session
- `isModerator()` - Check if user is moderator
- `isAdmin()` - Check if user is admin

### Moderation Store (`moderationStore.js`)

```javascript
const { pendingContent, pendingAppeals } = useModerationStore();
```

**State:**
- `pendingContent` - Content queue
- `pendingAppeals` - Appeal queue
- `currentContent` - Selected content

**Methods:**
- `setPendingContent(content)` - Update content list
- `addContent(content)` - Add new content
- `removeContent(id)` - Remove content
- `setContentLoading(bool)` - Loading state

## Component Guide

### Login Component
- Material-UI form with validation
- Demo login buttons for quick access
- Automatic redirect to dashboard
- Error handling

### Dashboard Component
- Content queue with filtering
- Status-based tabs
- Metrics cards
- Submit content dialog
- Navigation to content review

### ContentReview Component
- Full content display
- Toxicity score visualization
- Policy violations list
- Agent decision breakdown
- Moderation trigger button

### AppealsManagement Component
- Appeals table with filtering
- Review dialog with outcome selection
- Learning trigger warnings
- Automatic agent learning on overturn

### Analytics Component
- Line charts for learning trends
- Bar charts for agent performance
- Memory system statistics
- Filtering by agent and time range

## Customization

### Theme

Edit `App.jsx` to customize Material-UI theme:

```javascript
const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#dc004e' },
  },
});
```

### Routes

Add new routes in `App.jsx`:

```javascript
<Route path="/new-page" element={
  <ProtectedRoute>
    <NewComponent />
  </ProtectedRoute>
} />
```

## Development Tips

### Hot Reload
Vite provides instant hot module replacement. Changes appear immediately.

### API Debugging
Check browser DevTools > Network tab for API calls and responses.

### State Debugging
Install [Zustand DevTools](https://github.com/pmndrs/zustand#middleware) for state inspection.

### Component Debugging
Use React DevTools extension for component tree inspection.

## Troubleshooting

### Port Already in Use
```bash
# Change port in vite.config.js
server: { port: 3001 }
```

### API Connection Failed
- Ensure backend is running on `http://localhost:8002`
- Check `.env` for correct `VITE_API_URL`
- Verify CORS is enabled on backend

### Build Errors
```bash
rm -rf node_modules package-lock.json
npm install
```

### Login Not Working
- Check backend `/api/auth/login` endpoint
- Verify credentials match backend test users
- Check browser console for errors

## Performance Optimization

### Code Splitting
Routes are automatically code-split by React Router.

### Production Build
```bash
npm run build
# Output in dist/ folder
# Minified and optimized
```

### Asset Optimization
- Images should be under 200KB
- Use WebP format when possible
- Lazy load heavy components

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

When adding new features:
1. Follow existing component structure
2. Use Material-UI components
3. Add to appropriate route
4. Update API service if needed
5. Add to README

## License

Part of the Content Moderation Multi-Agent System.
Built for educational purposes (Udemy course).

## Support

For issues or questions:
- Check backend logs first
- Verify API endpoints are responding
- Review browser console errors
- Check network requests in DevTools
