"""
Authentication Database Module for Content Moderation Platform
Handles user management, authentication, and role-based access control for moderators
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path


class AuthDatabase:
    def __init__(self, db_path: str = "databases/moderation_auth.db"):
        # Convert relative path to absolute path relative to backend directory
        if not Path(db_path).is_absolute():
            backend_dir = Path(__file__).parent.parent.parent
            self.db_path = str(backend_dir / db_path)
        else:
            self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Create and return a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize the authentication database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Users table - for moderators and admins
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        ''')

        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Audit log table - tracks all moderator actions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                content_id TEXT,
                details TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Moderator assignments - track which moderators handle which content types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moderator_assignments (
                user_id INTEGER,
                content_type TEXT,
                priority_level TEXT,
                assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, content_type),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Moderator statistics - track performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moderator_stats (
                user_id INTEGER PRIMARY KEY,
                total_reviews INTEGER DEFAULT 0,
                approved_count INTEGER DEFAULT 0,
                removed_count INTEGER DEFAULT 0,
                warned_count INTEGER DEFAULT 0,
                escalated_count INTEGER DEFAULT 0,
                avg_response_time_seconds REAL DEFAULT 0,
                accuracy_score REAL DEFAULT 0,
                last_updated TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, password: str, full_name: str,
                   role: str, email: str = None, phone: str = None) -> bool:
        """Create a new user"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute('''
                INSERT INTO users (username, password_hash, full_name, role, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, full_name, role, email, phone))

            # Initialize moderator stats for new moderators
            user_id = cursor.lastrowid
            if role in ['moderator', 'senior_moderator', 'content_analyst', 'policy_specialist', 'admin']:
                cursor.execute('''
                    INSERT INTO moderator_stats (user_id, last_updated)
                    VALUES (?, ?)
                ''', (user_id, datetime.now().isoformat()))

            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            if conn:
                conn.close()

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user and return user details if successful"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute('''
                SELECT user_id, username, full_name, role, email, phone, is_active
                FROM users
                WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password_hash))

            user = cursor.fetchone()

            if user:
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = ? WHERE user_id = ?
                ''', (datetime.now().isoformat(), user['user_id']))
                conn.commit()

                user_dict = dict(user)
                return user_dict

            return None
        finally:
            if conn:
                conn.close()

    def create_session(self, user_id: int, ip_address: str = None) -> str:
        """Create a new session for a user"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            session_id = secrets.token_urlsafe(32)

            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id, ip_address)
                VALUES (?, ?, ?)
            ''', (session_id, user_id, ip_address))

            conn.commit()
            return session_id
        finally:
            if conn:
                conn.close()

    def get_session_user(self, session_id: str) -> Optional[Dict]:
        """Get user details from session ID"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT u.user_id, u.username, u.full_name, u.role, u.email, u.phone
                FROM users u
                JOIN user_sessions s ON u.user_id = s.user_id
                WHERE s.session_id = ? AND u.is_active = 1
            ''', (session_id,))

            user = cursor.fetchone()

            if user:
                return dict(user)
            return None
        finally:
            if conn:
                conn.close()

    def delete_session(self, session_id: str):
        """Delete a user session (logout)"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('DELETE FROM user_sessions WHERE session_id = ?', (session_id,))

            conn.commit()
        finally:
            if conn:
                conn.close()

    def log_action(self, user_id: int, action: str, content_id: str = None,
                   details: str = None, ip_address: str = None):
        """Log a moderator action for audit trail"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO audit_log (user_id, action, content_id, details, ip_address)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, action, content_id, details, ip_address))

            conn.commit()
        finally:
            if conn:
                conn.close()

    def get_audit_log(self, user_id: int = None, limit: int = 100) -> List[Dict]:
        """Get audit log entries, optionally filtered by user"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if user_id:
                cursor.execute('''
                    SELECT al.*, u.username, u.full_name
                    FROM audit_log al
                    JOIN users u ON al.user_id = u.user_id
                    WHERE al.user_id = ?
                    ORDER BY al.timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT al.*, u.username, u.full_name
                    FROM audit_log al
                    JOIN users u ON al.user_id = u.user_id
                    ORDER BY al.timestamp DESC
                    LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]
        finally:
            if conn:
                conn.close()

    def update_moderator_stats(self, user_id: int, action: str, response_time_seconds: float = None):
        """Update moderator performance statistics"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get current stats
            cursor.execute('SELECT * FROM moderator_stats WHERE user_id = ?', (user_id,))
            stats = cursor.fetchone()

            if stats:
                total_reviews = stats['total_reviews'] + 1
                approved_count = stats['approved_count'] + (1 if action == 'approve' else 0)
                removed_count = stats['removed_count'] + (1 if action == 'remove' else 0)
                warned_count = stats['warned_count'] + (1 if action == 'warn' else 0)
                escalated_count = stats['escalated_count'] + (1 if action == 'escalate' else 0)

                # Update average response time
                if response_time_seconds:
                    current_avg = stats['avg_response_time_seconds'] or 0
                    new_avg = ((current_avg * stats['total_reviews']) + response_time_seconds) / total_reviews
                else:
                    new_avg = stats['avg_response_time_seconds']

                cursor.execute('''
                    UPDATE moderator_stats
                    SET total_reviews = ?, approved_count = ?, removed_count = ?,
                        warned_count = ?, escalated_count = ?, avg_response_time_seconds = ?,
                        last_updated = ?
                    WHERE user_id = ?
                ''', (total_reviews, approved_count, removed_count, warned_count,
                      escalated_count, new_avg, datetime.now().isoformat(), user_id))
            else:
                # Create new stats record
                cursor.execute('''
                    INSERT INTO moderator_stats (user_id, total_reviews, approved_count,
                        removed_count, warned_count, escalated_count, avg_response_time_seconds, last_updated)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                ''', (user_id,
                      1 if action == 'approve' else 0,
                      1 if action == 'remove' else 0,
                      1 if action == 'warn' else 0,
                      1 if action == 'escalate' else 0,
                      response_time_seconds or 0,
                      datetime.now().isoformat()))

            conn.commit()
        finally:
            if conn:
                conn.close()

    def get_moderator_stats(self, user_id: int) -> Optional[Dict]:
        """Get performance statistics for a moderator"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT ms.*, u.username, u.full_name, u.role
                FROM moderator_stats ms
                JOIN users u ON ms.user_id = u.user_id
                WHERE ms.user_id = ?
            ''', (user_id,))

            stats = cursor.fetchone()
            return dict(stats) if stats else None
        finally:
            if conn:
                conn.close()

    def get_all_moderator_stats(self) -> List[Dict]:
        """Get performance statistics for all moderators"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT ms.*, u.username, u.full_name, u.role
                FROM moderator_stats ms
                JOIN users u ON ms.user_id = u.user_id
                ORDER BY ms.total_reviews DESC
            ''')

            return [dict(row) for row in cursor.fetchall()]
        finally:
            if conn:
                conn.close()

    def assign_moderator_to_content_type(self, user_id: int, content_type: str, priority_level: str = "normal"):
        """Assign a moderator to handle specific content types"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO moderator_assignments (user_id, content_type, priority_level, assigned_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, content_type, priority_level, datetime.now().isoformat()))

            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn:
                conn.close()

    def get_moderator_assignments(self, user_id: int) -> List[Dict]:
        """Get content type assignments for a moderator"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM moderator_assignments WHERE user_id = ?
            ''', (user_id,))

            return [dict(row) for row in cursor.fetchall()]
        finally:
            if conn:
                conn.close()

    def get_all_users(self) -> List[Dict]:
        """Get all users (for admin purposes)"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, username, full_name, role, email, phone, is_active, created_at, last_login
                FROM users
                ORDER BY created_at DESC
            ''')

            users = [dict(row) for row in cursor.fetchall()]
            return users
        finally:
            if conn:
                conn.close()

    def get_users_by_role(self, role: str) -> List[Dict]:
        """Get all users with a specific role"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, username, full_name, role, email, phone, is_active, created_at, last_login
                FROM users
                WHERE role = ?
                ORDER BY created_at DESC
            ''', (role,))

            return [dict(row) for row in cursor.fetchall()]
        finally:
            if conn:
                conn.close()

    def update_user_active_status(self, user_id: int, is_active: bool):
        """Activate or deactivate a user"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE users SET is_active = ? WHERE user_id = ?
            ''', (1 if is_active else 0, user_id))

            conn.commit()
        finally:
            if conn:
                conn.close()

    def change_password(self, user_id: int, new_password: str) -> bool:
        """Change user password"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            password_hash = self.hash_password(new_password)

            cursor.execute('''
                UPDATE users SET password_hash = ? WHERE user_id = ?
            ''', (password_hash, user_id))

            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn:
                conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, username, full_name, role, email, phone, is_active, created_at, last_login
                FROM users
                WHERE user_id = ?
            ''', (user_id,))

            user = cursor.fetchone()
            return dict(user) if user else None
        finally:
            if conn:
                conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, username, full_name, role, email, phone, is_active, created_at, last_login
                FROM users
                WHERE username = ?
            ''', (username,))

            user = cursor.fetchone()
            return dict(user) if user else None
        finally:
            if conn:
                conn.close()

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            password_hash = self.hash_password(new_password)

            cursor.execute('''
                UPDATE users
                SET password_hash = ?
                WHERE user_id = ?
            ''', (password_hash, user_id))

            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating password: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def update_user(self, user_id: int, full_name: str = None, role: str = None,
                   email: str = None, phone: str = None, is_active: bool = None) -> bool:
        """Update user information (admin only)"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Build dynamic update query
            updates = []
            params = []

            if full_name is not None:
                updates.append("full_name = ?")
                params.append(full_name)

            if role is not None:
                updates.append("role = ?")
                params.append(role)

            if email is not None:
                updates.append("email = ?")
                params.append(email)

            if phone is not None:
                updates.append("phone = ?")
                params.append(phone)

            if is_active is not None:
                updates.append("is_active = ?")
                params.append(1 if is_active else 0)

            if not updates:
                return True  # Nothing to update

            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"

            cursor.execute(query, params)
            conn.commit()

            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating user: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def delete_user(self, user_id: int) -> bool:
        """Delete a user (admin only)"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Delete user sessions first
            cursor.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))

            # Delete user
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))

            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
        finally:
            if conn:
                conn.close()