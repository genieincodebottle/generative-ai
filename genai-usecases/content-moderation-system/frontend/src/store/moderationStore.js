import { create } from 'zustand';

export const useModerationStore = create((set, get) => ({
  // Content queue
  pendingContent: [],
  currentContent: null,
  contentLoading: false,
  contentError: null,

  // Appeals
  pendingAppeals: [],
  appealLoading: false,
  appealError: null,

  // Filters
  statusFilter: 'all',
  sortBy: 'created_at',
  sortOrder: 'desc',

  // Actions - Content
  setPendingContent: (content) => set({ pendingContent: content }),

  setCurrentContent: (content) => set({ currentContent: content }),

  setContentLoading: (loading) => set({ contentLoading: loading }),

  setContentError: (error) => set({ contentError: error }),

  addContent: (content) => set((state) => ({
    pendingContent: [content, ...state.pendingContent],
  })),

  removeContent: (contentId) => set((state) => ({
    pendingContent: state.pendingContent.filter((c) => c.content_id !== contentId),
  })),

  updateContentStatus: (contentId, status) => set((state) => ({
    pendingContent: state.pendingContent.map((c) =>
      c.content_id === contentId ? { ...c, status } : c
    ),
  })),

  // Actions - Appeals
  setPendingAppeals: (appeals) => set({ pendingAppeals: appeals }),

  setAppealLoading: (loading) => set({ appealLoading: loading }),

  setAppealError: (error) => set({ appealError: error }),

  addAppeal: (appeal) => set((state) => ({
    pendingAppeals: [appeal, ...state.pendingAppeals],
  })),

  removeAppeal: (appealId) => set((state) => ({
    pendingAppeals: state.pendingAppeals.filter((a) => a.appeal_id !== appealId),
  })),

  updateAppealStatus: (appealId, status) => set((state) => ({
    pendingAppeals: state.pendingAppeals.map((a) =>
      a.appeal_id === appealId ? { ...a, status } : a
    ),
  })),

  // Actions - Filters
  setStatusFilter: (filter) => set({ statusFilter: filter }),

  setSortBy: (sortBy) => set({ sortBy }),

  setSortOrder: (order) => set({ sortOrder: order }),

  // Computed
  getFilteredContent: () => {
    const { pendingContent, statusFilter, sortBy, sortOrder } = get();
    let filtered = [...pendingContent];

    // Filter by status
    if (statusFilter !== 'all') {
      filtered = filtered.filter((c) => c.status === statusFilter);
    }

    // Sort
    filtered.sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      const comparison = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  },

  // Clear all
  clearAll: () => set({
    pendingContent: [],
    currentContent: null,
    pendingAppeals: [],
    contentError: null,
    appealError: null,
  }),
}));
