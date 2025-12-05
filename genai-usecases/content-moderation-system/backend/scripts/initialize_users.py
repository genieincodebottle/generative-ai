"""
Initialize Default Users for Content Moderation Platform
Creates default moderator users for testing and demo purposes
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.auth_db import AuthDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def initialize_default_users():
    """Create default users for the content moderation system"""

    db = AuthDatabase()

    logger.info("Initializing Content Moderation User Database...")

    # Define default users for content moderation platform
    # Roles aligned with content moderation workflow:
    # - user: End user who submits content
    # - moderator: Basic content review
    # - senior_moderator: Handle escalations and HITL queue
    # - content_analyst: Analyze patterns and trends
    # - policy_specialist: Handle policy violations and appeals
    # - admin: Full system access

    default_users = [
        # End Users (Content Submitters)
        {
            "username": "raj",
            "password": "test@123",
            "full_name": "Raj Singh",
            "role": "user",
            "email": "raj@example.com",
            "phone": "+919876543201"
        },
        {
            "username": "priya",
            "password": "test@123",
            "full_name": "Priya Sharma",
            "role": "user",
            "email": "priya@example.com",
            "phone": "+919876543202"
        },
        {
            "username": "amit",
            "password": "test@123",
            "full_name": "Amit Kumar",
            "role": "user",
            "email": "amit@example.com",
            "phone": "+919876543203"
        },
        # Basic Moderators
        {
            "username": "moderator1",
            "password": "mod@123",
            "full_name": "Vikram Patel - Content Moderator",
            "role": "moderator",
            "email": "vikram.patel@moderation.com",
            "phone": "+919876543101"
        },
        {
            "username": "moderator2",
            "password": "mod@123",
            "full_name": "Anjali Reddy - Content Moderator",
            "role": "moderator",
            "email": "anjali.reddy@moderation.com",
            "phone": "+919876543102"
        },
        # Senior Moderators (handle HITL escalations)
        {
            "username": "senior_mod",
            "password": "senior@123",
            "full_name": "Rahul Mehta - Senior Moderator",
            "role": "senior_moderator",
            "email": "rahul.mehta@moderation.com",
            "phone": "+919876543201"
        },
        {
            "username": "hitl_reviewer",
            "password": "hitl@123",
            "full_name": "Neha Gupta - HITL Review Specialist",
            "role": "senior_moderator",
            "email": "neha.gupta@moderation.com",
            "phone": "+919876543202"
        },
        # Content Analysts
        {
            "username": "analyst",
            "password": "analyst@123",
            "full_name": "Arjun Verma - Content Analyst",
            "role": "content_analyst",
            "email": "arjun.verma@moderation.com",
            "phone": "+919876543301"
        },
        # Policy Specialists (handle appeals and policy questions)
        {
            "username": "policy_expert",
            "password": "policy@123",
            "full_name": "Kavya Iyer - Policy Specialist",
            "role": "policy_specialist",
            "email": "kavya.iyer@moderation.com",
            "phone": "+919876543401"
        },
        {
            "username": "appeals_handler",
            "password": "appeals@123",
            "full_name": "Rohan Desai - Appeals Handler",
            "role": "policy_specialist",
            "email": "rohan.desai@moderation.com",
            "phone": "+919876543402"
        },
        # Administrator
        {
            "username": "admin",
            "password": "admin@123",
            "full_name": "Sanjay Kapoor - System Administrator",
            "role": "admin",
            "email": "sanjay.kapoor@moderation.com",
            "phone": "+919876543001"
        }
    ]

    # Create users
    created_count = 0
    skipped_count = 0

    for user_data in default_users:
        success = db.create_user(
            username=user_data["username"],
            password=user_data["password"],
            full_name=user_data["full_name"],
            role=user_data["role"],
            email=user_data["email"],
            phone=user_data["phone"]
        )

        if success:
            logger.info(f"[+] Created user: {user_data['username']} ({user_data['role']})")
            created_count += 1
        else:
            logger.info(f"[-] User already exists: {user_data['username']}")
            skipped_count += 1

    logger.info(f"Initialization complete!")
    logger.info(f"Created: {created_count} users")
    logger.info(f"Skipped: {skipped_count} users (already exist)")
    logger.info("DEFAULT LOGIN CREDENTIALS")
    logger.info("END USERS (Content Submitters):")
    logger.info("  - Username: raj           / Password: test@123      (Raj Singh)")
    logger.info("  - Username: priya         / Password: test@123      (Priya Sharma)")
    logger.info("  - Username: amit          / Password: test@123      (Amit Kumar)")
    logger.info("Content Moderators (Basic Review):")
    logger.info("  - Username: moderator1    / Password: mod@123       (Vikram Patel)")
    logger.info("  - Username: moderator2    / Password: mod@123       (Anjali Reddy)")
    logger.info("Senior Moderators (HITL & Escalations):")
    logger.info("  - Username: senior_mod    / Password: senior@123    (Rahul Mehta)")
    logger.info("  - Username: hitl_reviewer / Password: hitl@123      (Neha Gupta)")
    logger.info("Content Analyst (Patterns & Trends):")
    logger.info("  - Username: analyst       / Password: analyst@123   (Arjun Verma)")
    logger.info("Policy Specialists (Appeals & Violations):")
    logger.info("  - Username: policy_expert   / Password: policy@123  (Kavya Iyer)")
    logger.info("  - Username: appeals_handler / Password: appeals@123 (Rohan Desai)")
    logger.info("Administrator (Full Access):")
    logger.info("  - Username: admin         / Password: admin@123     (Sanjay Kapoor)")
    logger.info("ROLE PERMISSIONS:")
    logger.info("user             : Submit content, view status, file appeals")
    logger.info("moderator        : Review content, approve/warn/remove, basic actions")
    logger.info("senior_moderator : All moderator + HITL queue, escalations, user suspend")
    logger.info("content_analyst  : View analytics, patterns, trends, export reports")
    logger.info("policy_specialist: Handle appeals, policy violations, user bans")
    logger.info("admin            : Full system access, user management, configuration")


if __name__ == "__main__":
    initialize_default_users()
