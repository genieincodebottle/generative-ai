export interface Analytics {
  topics: string[];
  sentiment: string[];
  trends: string[];
}

export interface FeedbackItem {
  feedback: string;
  timestamp: string;
  username: string;
}

export interface FeedbackData {
  current: FeedbackItem[];
  historical: FeedbackItem[];
}

export enum UserRole {
  USER = "user",
  ADMIN = "admin",
}

export const API_BASE_URL = "http://localhost:8000";
