"""
Database Cleanup Script for Content Moderation System
Clears ALL data including user accounts.

Usage: python scripts/cleanup_data.py
"""

import sqlite3
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cleanup_moderation_data():
    """Clean up moderation database (content, appeals, reviews, etc.)"""
    db_path = "databases/moderation_data.db"

    if not os.path.exists(db_path):
        logger.info(f"Database {db_path} not found. Nothing to clean.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logger.info("CONTENT MODERATION DATABASE CLEANUP")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Tables to clean - ALL tables
    tables_to_clean = [
        "content_submissions",
        "agent_executions",
        "final_decisions",
        "appeals",
        "manual_reviews",
        "user_profiles",
        "moderation_history",
        "stories",
        "story_comments",
    ]

    for table in tables_to_clean:
        try:
            # Get count before deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            if count > 0:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"Cleared {table}: {count} records deleted")
            else:
                logger.info(f"- {table}: already empty")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                logger.info(f"{table}: table doesn't exist (skipped)")
            else:
                logger.info(f"{table}: Error - {e}")

    conn.commit()
    conn.close()


def cleanup_auth_database():
    """Clean up authentication database (users, sessions)"""
    db_path = "databases/moderation_auth.db"

    if not os.path.exists(db_path):
        logger.info(f"Database {db_path} not found. Nothing to clean.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    logger.info("=" * 40)
    logger.info("AUTHENTICATION DATABASE CLEANUP")
    logger.info("=" * 40)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Tables to clean
    tables_to_clean = [
        "users",
        "user_sessions",
    ]

    for table in tables_to_clean:
        try:
            # Get count before deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            if count > 0:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"Cleared {table}: {count} records deleted")
            else:
                logger.info(f"{table}: already empty")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                logger.error(f"{table}: table doesn't exist (skipped)")
            else:
                logger.error(f"{table}: Error - {e}")

    conn.commit()
    conn.close()


def cleanup_chroma_db():
    """Clean up ChromaDB vector store if it exists"""
    chroma_path = "databases/chroma_moderation_db"

    logger.info("=" * 40)
    logger.info("CHROMADB VECTOR STORE CLEANUP")
    logger.info("=" * 40)

    if os.path.exists(chroma_path):
        import shutil
        try:
            shutil.rmtree(chroma_path)
            logger.info(f"Cleared ChromaDB: {chroma_path}")
        except Exception as e:
            logger.error(f"ChromaDB cleanup error: {e}")
    else:
        logger.info(f"ChromaDB: not found (skipped)")


def main():
    logger.info("CONTENT MODERATION - FULL DATABASE CLEANUP")
    logger.info("Removes ALL Data Including Users")

    # Confirm before proceeding
    response = input("WARNING: This will DELETE ALL DATA:\n"
                     "   - All content, stories, comments\n"
                     "   - All appeals and reviews\n"
                     "   - All user accounts and sessions\n"
                     "   - ChromaDB vector store\n\n"
                     "   This action CANNOT be undone!\n\n"
                     "   Type 'DELETE ALL' to confirm: ").strip()

    if response != "DELETE ALL":
        logger.info("\nCleanup cancelled.")
        return

    # Cleanup moderation data
    cleanup_moderation_data()

    # Cleanup authentication database
    cleanup_auth_database()

    # Cleanup ChromaDB
    cleanup_chroma_db()

    logger.info("CLEANUP COMPLETE - ALL DATA REMOVED")
    logger.info("Next steps:")
    logger.info("  1. Run 'python scripts/initialize_users.py' to create default users")
    logger.info("  2. Restart the backend server")


if __name__ == "__main__":
    main()
