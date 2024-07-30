// types.ts
export interface Analytics {
    topics: string[];
    sentiment: string[];
    trends: string[];
  }
export enum UserRole {
  USER = "user",
  ADMIN = "admin"
}
