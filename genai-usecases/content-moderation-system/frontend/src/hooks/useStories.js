import { useState, useEffect, useCallback } from 'react';
import { storiesAPI } from '../services/api';

/**
 * Custom hook for fetching and managing stories data
 * Provides caching, error handling, and automatic refetching
 *
 * @param {Object} options - Configuration options
 * @param {boolean} options.visibleOnly - Fetch only visible stories
 * @param {number} options.limit - Max number of stories to fetch
 * @param {boolean} options.autoRefetch - Auto-refetch on mount
 * @param {number} options.refetchInterval - Refetch interval in ms
 * @returns {Object} Stories data, loading state, error, and refetch function
 */
export const useStories = (options = {}) => {
  const {
    visibleOnly = true,
    limit = 100,
    autoRefetch = false,
    refetchInterval = null
  } = options;

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastFetch, setLastFetch] = useState(null);

  const fetchStories = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await storiesAPI.getStories(visibleOnly, limit);
      console.log('useStories - API response:', response);
      const stories = response.stories || [];
      console.log('useStories - Extracted stories:', stories);

      setData({
        stories,
        count: stories.length,
        timestamp: new Date().toISOString()
      });
      setLastFetch(Date.now());
    } catch (err) {
      console.error('Error fetching stories:', err);
      setError(err.message || 'Failed to fetch stories');
    } finally {
      setLoading(false);
    }
  }, [visibleOnly, limit]);

  // Initial fetch
  useEffect(() => {
    fetchStories();
  }, [fetchStories]);

  // Auto-refetch interval
  useEffect(() => {
    if (autoRefetch && refetchInterval) {
      const interval = setInterval(fetchStories, refetchInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefetch, refetchInterval, fetchStories]);

  return {
    stories: data?.stories || [],
    count: data?.count || 0,
    loading,
    error,
    lastFetch,
    refetch: fetchStories
  };
};

/**
 * Hook for fetching user's stories
 */
export const useUserStories = (userId, options = {}) => {
  const { limit = 50 } = options;
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchUserStories = useCallback(async () => {
    if (!userId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await storiesAPI.getUserStories(userId, limit);
      const stories = response.stories || [];

      setData({
        stories,
        count: stories.length
      });
    } catch (err) {
      console.error('Error fetching user stories:', err);
      setError(err.message || 'Failed to fetch user stories');
    } finally {
      setLoading(false);
    }
  }, [userId, limit]);

  useEffect(() => {
    fetchUserStories();
  }, [fetchUserStories]);

  return {
    stories: data?.stories || [],
    count: data?.count || 0,
    loading,
    error,
    refetch: fetchUserStories
  };
};

/**
 * Hook for story statistics and analytics
 */
export const useStoryStats = (stories = []) => {
  const [stats, setStats] = useState({
    totalStories: 0,
    totalViews: 0,
    totalComments: 0,
    averageEngagement: 0,
    topStory: null,
    trending: []
  });

  useEffect(() => {
    if (!stories.length) {
      setStats({
        totalStories: 0,
        totalViews: 0,
        totalComments: 0,
        averageEngagement: 0,
        topStory: null,
        trending: []
      });
      return;
    }

    // Calculate statistics
    const totalViews = stories.reduce((sum, s) => sum + (s.view_count || 0), 0);
    const totalComments = stories.reduce((sum, s) => sum + (s.comment_count || 0), 0);
    const avgEngagement = (totalViews + totalComments) / stories.length;

    // Find top story
    const topStory = [...stories].sort(
      (a, b) => (b.view_count + b.comment_count) - (a.view_count + a.comment_count)
    )[0];

    // Calculate trending (high engagement + recent)
    const trending = [...stories]
      .filter(s => {
        const hoursSinceCreation = (Date.now() - new Date(s.created_at)) / (1000 * 60 * 60);
        return hoursSinceCreation < 48; // Last 48 hours
      })
      .sort((a, b) =>
        (b.view_count + b.comment_count * 3) - (a.view_count + a.comment_count * 3)
      )
      .slice(0, 5);

    setStats({
      totalStories: stories.length,
      totalViews,
      totalComments,
      averageEngagement: Math.round(avgEngagement),
      topStory,
      trending
    });
  }, [stories]);

  return stats;
};

/**
 * Hook for featured stories calculation
 */
export const useFeaturedStories = (stories = [], count = 3) => {
  const [featured, setFeatured] = useState([]);

  useEffect(() => {
    if (!stories.length) {
      setFeatured([]);
      return;
    }

    // Calculate featured based on engagement score
    const featuredStories = [...stories]
      .map(story => ({
        ...story,
        engagementScore: (story.view_count || 0) + (story.comment_count || 0) * 2
      }))
      .sort((a, b) => b.engagementScore - a.engagementScore)
      .slice(0, count);

    setFeatured(featuredStories);
  }, [stories, count]);

  return featured;
};

/**
 * Hook for searching and filtering stories
 */
export const useStoriesFilter = (stories = [], searchQuery = '', filters = {}) => {
  const [filtered, setFiltered] = useState([]);

  useEffect(() => {
    let result = [...stories];

    // Apply search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(story =>
        story.title?.toLowerCase().includes(query) ||
        story.content_text?.toLowerCase().includes(query) ||
        story.username?.toLowerCase().includes(query)
      );
    }

    // Apply filters
    if (filters.sortBy) {
      switch (filters.sortBy) {
        case 'trending':
          result.sort((a, b) =>
            (b.view_count + b.comment_count * 3) - (a.view_count + a.comment_count * 3)
          );
          break;
        case 'newest':
          result.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
          break;
        case 'mostViewed':
          result.sort((a, b) => (b.view_count || 0) - (a.view_count || 0));
          break;
        case 'mostCommented':
          result.sort((a, b) => (b.comment_count || 0) - (a.comment_count || 0));
          break;
        default:
          break;
      }
    }

    if (filters.status) {
      result = result.filter(story => story.moderation_status === filters.status);
    }

    setFiltered(result);
  }, [stories, searchQuery, filters]);

  return filtered;
};
