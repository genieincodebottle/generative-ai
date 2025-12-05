"""
Database module for content moderation system using SQLite.

Stores:
- Content submissions
- Moderation decisions
- User profiles
- Agent execution details
- Manual reviews
- Appeal records
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path


class ModerationDatabase:
    """SQLite database for storing moderation data."""

    def __init__(self, db_path: str = "databases/moderation_data.db"):
        """
        Initialize the database connection.

        Args:
            db_path: Path to SQLite database file
        """
        # Convert relative path to absolute path relative to backend directory
        if not Path(db_path).is_absolute():
            backend_dir = Path(__file__).parent.parent.parent
            self.db_path = str(backend_dir / db_path)
        else:
            self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Content submissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT UNIQUE NOT NULL,
                    submission_id TEXT,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content_text TEXT,
                    content_type TEXT NOT NULL,
                    platform TEXT,
                    language TEXT DEFAULT 'en',
                    submission_timestamp TEXT NOT NULL,
                    current_status TEXT NOT NULL,
                    moderation_action TEXT,
                    action_reason TEXT,
                    toxicity_score REAL DEFAULT 0.0,
                    violation_severity TEXT,
                    requires_human_review INTEGER DEFAULT 0,
                    content_removed INTEGER DEFAULT 0,
                    user_notified INTEGER DEFAULT 0,
                    processed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    username TEXT NOT NULL,
                    account_age_days INTEGER DEFAULT 0,
                    total_posts INTEGER DEFAULT 0,
                    total_violations INTEGER DEFAULT 0,
                    previous_warnings INTEGER DEFAULT 0,
                    previous_suspensions INTEGER DEFAULT 0,
                    reputation_score REAL DEFAULT 0.7,
                    reputation_tier TEXT DEFAULT 'new_user',
                    verified INTEGER DEFAULT 0,
                    follower_count INTEGER DEFAULT 0,
                    is_suspended INTEGER DEFAULT 0,
                    is_banned INTEGER DEFAULT 0,
                    suspension_until TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Agent executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    flags TEXT,
                    recommendations TEXT,
                    extracted_data TEXT,
                    requires_human_review INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    execution_order INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (content_id) REFERENCES content_submissions(content_id)
                )
            """)

            # Policy violations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policy_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detected_by_agent TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (content_id) REFERENCES content_submissions(content_id)
                )
            """)

            # Manual reviews table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS manual_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    reviewer_name TEXT NOT NULL,
                    review_decision TEXT NOT NULL,
                    review_notes TEXT,
                    previous_status TEXT,
                    new_status TEXT,
                    review_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (content_id) REFERENCES content_submissions(content_id)
                )
            """)

            # Appeals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appeals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    appeal_reason TEXT NOT NULL,
                    original_decision TEXT NOT NULL,
                    appeal_decision TEXT,
                    appeal_reasoning TEXT,
                    appeal_timestamp TEXT NOT NULL,
                    decision_timestamp TEXT,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (content_id) REFERENCES content_submissions(content_id)
                )
            """)

            # User actions table (suspensions, bans, warnings)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    reason TEXT,
                    content_id TEXT,
                    duration_days INTEGER,
                    action_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)

            # Stories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    content_id TEXT,
                    moderation_status TEXT DEFAULT 'pending',
                    is_approved INTEGER DEFAULT 0,
                    is_visible INTEGER DEFAULT 0,
                    toxicity_score REAL DEFAULT 0.0,
                    view_count INTEGER DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)

            # Story comments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS story_comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_id TEXT UNIQUE NOT NULL,
                    story_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    content_id TEXT,
                    moderation_status TEXT DEFAULT 'pending',
                    is_approved INTEGER DEFAULT 0,
                    is_visible INTEGER DEFAULT 0,
                    toxicity_score REAL DEFAULT 0.0,
                    parent_comment_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (story_id) REFERENCES stories(story_id),
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_user ON content_submissions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_status ON content_submissions(current_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_profiles(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_content ON agent_executions(content_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_violations_content ON policy_violations(content_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_story_user ON stories(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_story_status ON stories(moderation_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_story_visible ON stories(is_visible)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comment_story ON story_comments(story_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comment_user ON story_comments(user_id)")

            print(f"✅ Database initialized at {self.db_path}")

    def create_content_submission(self, content_data: Dict[str, Any]) -> str:
        """
        Create a new content submission record.

        Args:
            content_data: Dictionary with content information

        Returns:
            content_id of created submission
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO content_submissions (
                    content_id, submission_id, user_id, username, content_text,
                    content_type, platform, language, submission_timestamp,
                    current_status, toxicity_score, requires_human_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_data.get("content_id"),
                content_data.get("submission_id"),
                content_data.get("user_id"),
                content_data.get("username"),
                content_data.get("content_text"),
                content_data.get("content_type"),
                content_data.get("platform", "generic"),
                content_data.get("language", "en"),
                content_data.get("submission_timestamp"),
                content_data.get("status", "submitted"),
                content_data.get("toxicity_score", 0.0),
                1 if content_data.get("requires_human_review", False) else 0
            ))

            return content_data.get("content_id")

    def update_content_status(
        self,
        content_id: str,
        status: str,
        moderation_action: Optional[str] = None,
        action_reason: Optional[str] = None,
        toxicity_score: Optional[float] = None
    ):
        """Update content submission status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE content_submissions
                SET current_status = ?,
                    moderation_action = COALESCE(?, moderation_action),
                    action_reason = COALESCE(?, action_reason),
                    toxicity_score = COALESCE(?, toxicity_score),
                    processed_at = ?
                WHERE content_id = ?
            """, (status, moderation_action, action_reason, toxicity_score, datetime.now().isoformat(), content_id))

    def save_agent_decision(self, content_id: str, agent_decision: Any):
        """Save an agent's decision to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get current execution order
            cursor.execute("""
                SELECT COALESCE(MAX(execution_order), 0) + 1
                FROM agent_executions
                WHERE content_id = ?
            """, (content_id,))

            execution_order = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO agent_executions (
                    content_id, agent_name, decision, confidence, reasoning,
                    flags, recommendations, extracted_data, requires_human_review,
                    processing_time, execution_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_id,
                agent_decision.agent_name,
                agent_decision.decision.value,
                agent_decision.confidence,
                agent_decision.reasoning,
                json.dumps(agent_decision.flags),
                json.dumps(agent_decision.recommendations),
                json.dumps(agent_decision.extracted_data),
                1 if agent_decision.requires_human_review else 0,
                agent_decision.processing_time,
                execution_order
            ))

    def save_policy_violations(self, content_id: str, violations: List[str], severity: str, agent_name: str):
        """Save policy violations for a content submission."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            for violation in violations:
                cursor.execute("""
                    INSERT INTO policy_violations (
                        content_id, violation_type, severity, detected_by_agent
                    ) VALUES (?, ?, ?, ?)
                """, (content_id, violation, severity, agent_name))

    def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve content submission by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM content_submissions WHERE content_id = ?
            """, (content_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_content(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all content submissions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM content_submissions
                ORDER BY submission_timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_content_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get content by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM content_submissions
                WHERE current_status = ?
                ORDER BY submission_timestamp DESC
                LIMIT ?
            """, (status, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_agent_executions(self, content_id: str) -> List[Dict[str, Any]]:
        """Get all agent executions for a content submission."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM agent_executions
                WHERE content_id = ?
                ORDER BY execution_order ASC
            """, (content_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_policy_violations(self, content_id: str) -> List[Dict[str, Any]]:
        """Get all policy violations for a content submission."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM policy_violations
                WHERE content_id = ?
                ORDER BY timestamp ASC
            """, (content_id,))

            return [dict(row) for row in cursor.fetchall()]

    def create_or_update_user(self, user_data: Dict[str, Any]):
        """Create or update user profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO user_profiles (
                    user_id, username, account_age_days, total_posts,
                    total_violations, previous_warnings, previous_suspensions,
                    reputation_score, reputation_tier, verified, follower_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    total_posts = total_posts + 1,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_data.get("user_id"),
                user_data.get("username"),
                user_data.get("account_age_days", 0),
                user_data.get("total_posts", 0),
                user_data.get("total_violations", 0),
                user_data.get("previous_warnings", 0),
                user_data.get("previous_suspensions", 0),
                user_data.get("reputation_score", 0.7),
                user_data.get("reputation_tier", "new_user"),
                1 if user_data.get("verified", False) else 0,
                user_data.get("follower_count", 0)
            ))

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
            """, (user_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_user_reputation(self, user_id: str, new_score: float, new_tier: str):
        """Update user reputation score."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_profiles
                SET reputation_score = ?,
                    reputation_tier = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (new_score, new_tier, user_id))

    def record_user_action(
        self,
        user_id: str,
        action_type: str,
        reason: str,
        content_id: Optional[str] = None,
        duration_days: Optional[int] = None
    ):
        """Record a user action (warning, suspension, ban)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            expires_at = None
            if duration_days:
                from datetime import timedelta
                expires_at = (datetime.now() + timedelta(days=duration_days)).isoformat()

            cursor.execute("""
                INSERT INTO user_actions (
                    user_id, action_type, reason, content_id, duration_days, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, action_type, reason, content_id, duration_days, expires_at))

            # Update user profile based on action
            if action_type == "warning":
                cursor.execute("""
                    UPDATE user_profiles
                    SET previous_warnings = previous_warnings + 1
                    WHERE user_id = ?
                """, (user_id,))
            elif action_type == "suspension":
                cursor.execute("""
                    UPDATE user_profiles
                    SET previous_suspensions = previous_suspensions + 1,
                        is_suspended = 1,
                        suspension_until = ?
                    WHERE user_id = ?
                """, (expires_at, user_id))
            elif action_type == "ban":
                cursor.execute("""
                    UPDATE user_profiles
                    SET is_banned = 1
                    WHERE user_id = ?
                """, (user_id,))

    def increment_user_violations(self, user_id: str):
        """Increment user's total violations count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_profiles
                SET total_violations = total_violations + 1
                WHERE user_id = ?
            """, (user_id,))

    def save_manual_review(
        self,
        content_id: str,
        reviewer_name: str,
        decision: str,
        notes: str
    ):
        """Save a manual review decision."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get current status
            cursor.execute("""
                SELECT current_status FROM content_submissions WHERE content_id = ?
            """, (content_id,))

            row = cursor.fetchone()
            previous_status = row[0] if row else "unknown"

            cursor.execute("""
                INSERT INTO manual_reviews (
                    content_id, reviewer_name, review_decision, review_notes,
                    previous_status, new_status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (content_id, reviewer_name, decision, notes, previous_status, decision))

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total content
            cursor.execute("SELECT COUNT(*) FROM content_submissions")
            stats["total_content"] = cursor.fetchone()[0]

            # By status
            cursor.execute("""
                SELECT current_status, COUNT(*) as count
                FROM content_submissions
                GROUP BY current_status
            """)
            stats["by_status"] = {row[0]: row[1] for row in cursor.fetchall()}

            # Total users
            cursor.execute("SELECT COUNT(*) FROM user_profiles")
            stats["total_users"] = cursor.fetchone()[0]

            # Total violations
            cursor.execute("SELECT COUNT(*) FROM policy_violations")
            stats["total_violations"] = cursor.fetchone()[0]

            # Total reviews
            cursor.execute("SELECT COUNT(*) FROM manual_reviews")
            stats["total_reviews"] = cursor.fetchone()[0]

            return stats

    def get_agent_decisions(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all agent decisions/executions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM agent_executions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_all_appeals(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all appeals."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM appeals
                ORDER BY appeal_timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID (alias for get_user_profile)."""
        return self.get_user_profile(user_id)

    def update_user_status(self, user_id: str, status: str, reason: str = ""):
        """Update user status (active, suspended, banned, restricted)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if status == "suspended":
                cursor.execute("""
                    UPDATE user_profiles
                    SET is_suspended = 1,
                        is_banned = 0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))
            elif status == "banned":
                cursor.execute("""
                    UPDATE user_profiles
                    SET is_banned = 1,
                        is_suspended = 0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))
            elif status == "active":
                cursor.execute("""
                    UPDATE user_profiles
                    SET is_suspended = 0,
                        is_banned = 0,
                        suspension_until = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))
            elif status == "restricted":
                # Just mark as suspended for now
                cursor.execute("""
                    UPDATE user_profiles
                    SET is_suspended = 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))

            # Record the action
            cursor.execute("""
                INSERT INTO user_actions (
                    user_id, action_type, reason
                ) VALUES (?, ?, ?)
            """, (user_id, f"status_change_{status}", reason))

    def get_user_actions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all actions taken on a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM user_actions
                WHERE user_id = ?
                ORDER BY action_timestamp DESC
            """, (user_id,))

            return [dict(row) for row in cursor.fetchall()]

    # ═══════════════════════════════════════════════════════════════════════════════
    # Story Methods
    # ═══════════════════════════════════════════════════════════════════════════════

    def create_story(self, story_data: Dict[str, Any]) -> str:
        """Create a new story."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO stories (
                    story_id, user_id, username, title, content_text,
                    content_id, moderation_status, is_approved, is_visible,
                    toxicity_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                story_data.get("story_id"),
                story_data.get("user_id"),
                story_data.get("username"),
                story_data.get("title"),
                story_data.get("content_text"),
                story_data.get("content_id"),
                story_data.get("moderation_status", "pending"),
                1 if story_data.get("is_approved", False) else 0,
                1 if story_data.get("is_visible", False) else 0,
                story_data.get("toxicity_score", 0.0),
                story_data.get("created_at", datetime.now().isoformat())
            ))

            return story_data.get("story_id")

    def get_story_by_id(self, story_id: str) -> Optional[Dict[str, Any]]:
        """Get a story by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM stories WHERE story_id = ?
            """, (story_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_story_by_content_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get a story by its moderation content_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM stories WHERE content_id = ?
            """, (content_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_comment_by_content_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get a comment by its moderation content_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM story_comments WHERE content_id = ?
            """, (content_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_stories(self, limit: int = 100, visible_only: bool = False) -> List[Dict[str, Any]]:
        """Get all stories, optionally only visible ones."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if visible_only:
                cursor.execute("""
                    SELECT * FROM stories
                    WHERE is_visible = 1
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT * FROM stories
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_user_stories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get stories by a specific user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM stories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))

            return [dict(row) for row in cursor.fetchall()]

    def update_story_moderation(
        self,
        story_id: str,
        moderation_status: str,
        is_approved: bool,
        is_visible: bool,
        toxicity_score: Optional[float] = None
    ):
        """Update story moderation status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE stories
                SET moderation_status = ?,
                    is_approved = ?,
                    is_visible = ?,
                    toxicity_score = COALESCE(?, toxicity_score),
                    updated_at = ?
                WHERE story_id = ?
            """, (
                moderation_status,
                1 if is_approved else 0,
                1 if is_visible else 0,
                toxicity_score,
                datetime.now().isoformat(),
                story_id
            ))

    def increment_story_view(self, story_id: str):
        """Increment story view count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE stories
                SET view_count = view_count + 1
                WHERE story_id = ?
            """, (story_id,))

    def increment_story_comments(self, story_id: str):
        """Increment story comment count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE stories
                SET comment_count = comment_count + 1
                WHERE story_id = ?
            """, (story_id,))

    # ═══════════════════════════════════════════════════════════════════════════════
    # Story Comment Methods
    # ═══════════════════════════════════════════════════════════════════════════════

    def create_story_comment(self, comment_data: Dict[str, Any]) -> str:
        """Create a new story comment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO story_comments (
                    comment_id, story_id, user_id, username, content_text,
                    content_id, moderation_status, is_approved, is_visible,
                    toxicity_score, parent_comment_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                comment_data.get("comment_id"),
                comment_data.get("story_id"),
                comment_data.get("user_id"),
                comment_data.get("username"),
                comment_data.get("content_text"),
                comment_data.get("content_id"),
                comment_data.get("moderation_status", "pending"),
                1 if comment_data.get("is_approved", False) else 0,
                1 if comment_data.get("is_visible", False) else 0,
                comment_data.get("toxicity_score", 0.0),
                comment_data.get("parent_comment_id"),
                comment_data.get("created_at", datetime.now().isoformat())
            ))

            return comment_data.get("comment_id")

    def get_story_comments(self, story_id: str, visible_only: bool = False) -> List[Dict[str, Any]]:
        """Get all comments for a story."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if visible_only:
                cursor.execute("""
                    SELECT * FROM story_comments
                    WHERE story_id = ? AND is_visible = 1
                    ORDER BY created_at ASC
                """, (story_id,))
            else:
                cursor.execute("""
                    SELECT * FROM story_comments
                    WHERE story_id = ?
                    ORDER BY created_at ASC
                """, (story_id,))

            return [dict(row) for row in cursor.fetchall()]

    def get_comment_by_id(self, comment_id: str) -> Optional[Dict[str, Any]]:
        """Get a comment by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM story_comments WHERE comment_id = ?
            """, (comment_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_comment_moderation(
        self,
        comment_id: str,
        moderation_status: str,
        is_approved: bool,
        is_visible: bool,
        toxicity_score: Optional[float] = None
    ):
        """Update comment moderation status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE story_comments
                SET moderation_status = ?,
                    is_approved = ?,
                    is_visible = ?,
                    toxicity_score = COALESCE(?, toxicity_score)
                WHERE comment_id = ?
            """, (
                moderation_status,
                1 if is_approved else 0,
                1 if is_visible else 0,
                toxicity_score,
                comment_id
            ))

    def get_pending_stories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get stories pending moderation (including flagged stories)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM stories
                WHERE moderation_status IN ('pending', 'under_review', 'flagged')
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_pending_comments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get comments pending moderation (including flagged comments)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM story_comments
                WHERE moderation_status IN ('pending', 'under_review', 'flagged')
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]
