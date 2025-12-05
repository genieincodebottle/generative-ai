import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Badge,
  Chip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Gavel as GavelIcon,
  BarChart as BarChartIcon,
  Logout as LogoutIcon,
  Person as PersonIcon,
  Shield,
  NotificationsActive,
  Article,
  SupervisorAccount,
  Policy,
  AdminPanelSettings,
  Home,
  AutoStories,
  ManageAccounts,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useAuthStore } from '../../store/authStore';
import { useModerationStore } from '../../store/moderationStore';
import { authAPI } from '../../services/api';

const drawerWidth = 260;

// Role configurations
const roleConfig = {
  user: {
    label: 'User',
    color: 'default',
    icon: <PersonIcon />,
  },
  moderator: {
    label: 'Moderator',
    color: 'primary',
    icon: <Article />,
  },
  senior_moderator: {
    label: 'Senior Mod',
    color: 'secondary',
    icon: <SupervisorAccount />,
  },
  content_analyst: {
    label: 'Analyst',
    color: 'info',
    icon: <BarChartIcon />,
  },
  policy_specialist: {
    label: 'Policy',
    color: 'warning',
    icon: <Policy />,
  },
  admin: {
    label: 'Admin',
    color: 'error',
    icon: <AdminPanelSettings />,
  },
};

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout, isModerator, isSeniorModerator, canReviewAppeals, canViewAnalytics } = useAuthStore();
  const { pendingContent, pendingAppeals } = useModerationStore();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      await authAPI.logout();
    } finally {
      logout();
      navigate('/login');
    }
  };

  // Build menu items based on user role
  const getMenuItems = () => {
    const items = [];

    // Community - Main dashboard for all logged in users
    items.push({
      text: 'Community',
      icon: <Home />,
      path: '/community',
    });

    // My Appeals - for all users (to submit and track their appeals)
    items.push({
      text: 'My Appeals',
      icon: <GavelIcon />,
      path: '/my-appeals',
    });

    // Dashboard - for all moderators
    if (isModerator()) {
      items.push({
        text: 'Moderation',
        icon: <DashboardIcon />,
        path: '/dashboard',
        badge: pendingContent.length,
        divider: true,
      });
    }

    // Appeals Management - for policy specialists and admins
    if (canReviewAppeals()) {
      items.push({
        text: 'Appeals Review',
        icon: <GavelIcon />,
        path: '/appeals',
        badge: pendingAppeals.length,
      });
    }

    // Analytics - for analysts and above
    if (canViewAnalytics()) {
      items.push({
        text: 'Analytics',
        icon: <BarChartIcon />,
        path: '/analytics',
      });
    }

    // User Management - for admins only
    if (user?.role === 'admin') {
      items.push({
        text: 'User Management',
        icon: <ManageAccounts />,
        path: '/user-management',
        divider: true,
      });
    }

    // Settings - for all users
    items.push({
      text: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings',
    });

    return items;
  };

  const menuItems = getMenuItems();
  const currentRole = roleConfig[user?.role] || roleConfig.user;

  const drawer = (
    <div>
      <Toolbar sx={{ flexDirection: 'column', alignItems: 'flex-start', py: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Shield sx={{ mr: 1, color: 'primary.main', fontSize: 28 }} />
          <Typography variant="h6" noWrap component="div">
            SafeGuard AI
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary">
          Content Moderation Platform
        </Typography>
      </Toolbar>
      <Divider />

      {/* User Info Section */}
      <Box sx={{ p: 2, bgcolor: 'grey.50' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Avatar sx={{ bgcolor: `${currentRole.color}.main`, width: 40, height: 40 }}>
            {user?.username?.charAt(0).toUpperCase()}
          </Avatar>
          <Box>
            <Typography variant="subtitle2">
              {user?.full_name || user?.username}
            </Typography>
            <Chip
              size="small"
              label={currentRole.label}
              color={currentRole.color}
              icon={currentRole.icon}
              sx={{ height: 22, '& .MuiChip-icon': { fontSize: 14 } }}
            />
          </Box>
        </Box>
      </Box>
      <Divider />

      <List>
        {menuItems.map((item, index) => (
          <React.Fragment key={item.text}>
            {item.divider && <Divider sx={{ my: 1 }} />}
            <ListItem disablePadding>
              <ListItemButton
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
                sx={{
                  mx: 1,
                  borderRadius: 2,
                  mb: 0.5,
                  '&.Mui-selected': {
                    bgcolor: 'primary.50',
                    '&:hover': {
                      bgcolor: 'primary.100',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 40 }}>
                  {item.badge ? (
                    <Badge badgeContent={item.badge} color={item.badgeColor || 'error'}>
                      {item.icon}
                    </Badge>
                  ) : (
                    item.icon
                  )}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          </React.Fragment>
        ))}
      </List>

      {/* Role Info at bottom */}
      <Box sx={{ position: 'absolute', bottom: 0, left: 0, right: 0, p: 2 }}>
        <Divider sx={{ mb: 2 }} />
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          Access Level:
        </Typography>
        <Typography variant="body2" fontWeight="bold">
          {currentRole.label}
        </Typography>
      </Box>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          bgcolor: 'white',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {location.pathname === '/community' && 'Community Dashboard'}
            {location.pathname === '/portal' && 'Community Content Portal'}
            {location.pathname.startsWith('/stories/') && 'Story Details'}
            {location.pathname === '/my-appeals' && 'My Appeals'}
            {location.pathname === '/dashboard' && 'Moderator Dashboard'}
            {location.pathname === '/hitl' && 'Human-in-the-Loop Queue'}
            {location.pathname === '/appeals' && 'Appeals Management'}
            {location.pathname === '/analytics' && 'Analytics & Learning'}
            {location.pathname.startsWith('/review/') && 'Content Review'}
            {location.pathname === '/user-management' && 'User Management'}
            {location.pathname === '/settings' && 'Account Settings'}
          </Typography>

          {/* Notifications */}
          {isModerator() && pendingContent.length > 0 && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mr: 2 }}>
              <Badge badgeContent={pendingContent.length} color="warning">
                <NotificationsActive color="action" />
              </Badge>
            </Box>
          )}

          <IconButton onClick={handleProfileMenuOpen} color="inherit">
            <Avatar sx={{ width: 36, height: 36, bgcolor: `${currentRole.color}.main` }}>
              {user?.username?.charAt(0).toUpperCase()}
            </Avatar>
          </IconButton>

          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleProfileMenuClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem disabled>
              <ListItemIcon>
                {currentRole.icon}
              </ListItemIcon>
              <Box>
                <Typography variant="body2">{user?.full_name || user?.username}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {currentRole.label}
                </Typography>
              </Box>
            </MenuItem>
            <Divider />
            <MenuItem onClick={handleLogout}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" />
              </ListItemIcon>
              Logout
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
          minHeight: '100vh',
          bgcolor: 'grey.50',
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
