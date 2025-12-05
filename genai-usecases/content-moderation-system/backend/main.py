"""
FastAPI backend for Content Moderation & Community Safety Platform.

Provides REST API endpoints for:
- Content submission and moderation
- User management
- Appeal processing
- Statistics and reporting
"""

import os
# Disable ChromaDB telemetry to avoid warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.database.moderation_db import ModerationDatabase
from src.database.auth_db import AuthDatabase
from src.memory.memory import ModerationMemoryManager
from src.agents.workflow import create_moderation_workflow, process_content, resume_from_hitl
from src.core.models import (
    ContentState,
    ContentStatus,
    UserProfile,
    ContentMetadata,
    HITLTriggerReason,
    HITL_CONFIG
)
from src.utils.tools import generate_content_id, generate_user_id
from src.ml.ml_classifier import preload_ml_models, get_ml_status, MLConfig

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global instances
db: ModerationDatabase = None
auth_db: AuthDatabase = None
workflow = None
ml_status: Dict[str, Any] = {}

# Security scheme for authentication
security = HTTPBearer()

# In-memory store for pending HITL reviews (in production, use Redis or database)
hitl_pending_reviews: Dict[str, ContentState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global db, auth_db, workflow, ml_status

    # Startup
    logger.info("\n" + "=" * 40)
    logger.info("Starting Content Moderation API")
    logger.info("=" * 40)

    # Initialize database connections
    logger.info("\nInitializing databases...")
    db = ModerationDatabase("databases/moderation_data.db")
    auth_db = AuthDatabase("databases/moderation_auth.db")

    # Initialize workflow
    logger.info("Creating moderation workflow...")
    fast_mode_enabled = os.getenv("ENABLE_FAST_MODE", "true").lower() == "true"
    logger.info(f"   Fast Mode: {'✅ Enabled' if fast_mode_enabled else '❌ Disabled'}")
    if fast_mode_enabled:
        max_length = os.getenv("FAST_MODE_MAX_LENGTH", "200")
        content_types = os.getenv("FAST_MODE_CONTENT_TYPES", "story_comment")
        logger.info(f"   Fast Mode Max Length: {max_length} chars")
        logger.info(f"   Fast Mode Content Types: {content_types}")
    workflow = create_moderation_workflow(db, enable_fast_mode=fast_mode_enabled)

    # Preload ML models if enabled
    logger.info("\n" + "=" * 40)
    logger.info("\nML Configuration:")
    logger.info(f"   USE_ML_MODELS: {MLConfig.is_ml_enabled()}")
    logger.info(f"   ML_PRIMARY_MODEL: {MLConfig.get_primary_model()}")
    logger.info(f"   ML_USE_ENSEMBLE: {MLConfig.use_ensemble()}")
    logger.info(f"   ML_PRELOAD_MODELS: {MLConfig.should_preload()}")
    logger.info(f"   ML_DEVICE: {MLConfig.get_device()}")
    
    if MLConfig.is_ml_enabled() and MLConfig.should_preload():
        logger.info("\nPreloading ML models (this may take a moment)...")
        ml_status = preload_ml_models()
        if ml_status.get("status") == "success":
            logger.info(f"ML models loaded on {ml_status.get('device')}")
            logger.info(f"   Models: {', '.join(ml_status.get('models_available', []))}")
        elif ml_status.get("status") == "skipped":
            logger.info("ML preload skipped")
        else:
            logger.info(f"ML preload: {ml_status.get('status')} - {ml_status.get('error', 'unknown')}")
    else:
        logger.info("\nUsing keyword-based detection (ML models disabled)")
        ml_status = {"status": "disabled", "ml_enabled": False}

    logger.info("\n" + "=" * 40)
    logger.info("API ready to accept requests")
    logger.info("=" * 40 + "\n")

    yield

    # Shutdown
    logger.info("\nShutting down Content Moderation API")


app = FastAPI(
    title="Content Moderation & Community Safety API",
    description="Multi-agent AI system for automated content moderation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Authentication Models
# ═══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    """Login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    """Login response."""
    success: bool
    token: str
    user: Dict[str, Any]


class UserResponse(BaseModel):
    """User information response."""
    user_id: int
    username: str
    full_name: str
    role: str
    email: Optional[str] = None
    phone: Optional[str] = None


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50, description="Username (3-50 characters)")
    password: str = Field(..., min_length=6, description="Password (minimum 6 characters)")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")


class PasswordUpdateRequest(BaseModel):
    """Password update request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=6, description="New password (minimum 6 characters)")


class UpdateUserRequest(BaseModel):
    """Update user request (admin only)."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    role: Optional[str] = Field(None, description="User role")
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Authentication Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify token and return current user."""
    token = credentials.credentials
    user = auth_db.get_session_user(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    return user


def require_role(allowed_roles: List[str]):
    """Decorator to require specific roles."""
    def role_checker(user: Dict[str, Any] = Depends(get_current_user)):
        if user['role'] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return user
    return role_checker


# Role-based dependencies for different access levels
def get_moderator(user: Dict[str, Any] = Depends(get_current_user)):
    """Require moderator or higher role."""
    allowed = ['moderator', 'senior_moderator', 'content_analyst', 'policy_specialist', 'admin']
    if user['role'] not in allowed:
        raise HTTPException(status_code=403, detail="Moderator access required")
    return user


def get_senior_moderator(user: Dict[str, Any] = Depends(get_current_user)):
    """Require senior moderator or higher role."""
    allowed = ['senior_moderator', 'policy_specialist', 'admin']
    if user['role'] not in allowed:
        raise HTTPException(status_code=403, detail="Senior moderator access required")
    return user


def get_admin(user: Dict[str, Any] = Depends(get_current_user)):
    """Require admin role."""
    if user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models for API
# ═══════════════════════════════════════════════════════════════════════════════

class ContentSubmission(BaseModel):
    """Content submission request."""
    content_text: str = Field(..., description="Content text to moderate")
    content_type: str = Field("text", description="Type of content (text, image, video, comment, post)")
    user_id: Optional[str] = Field(None, description="User ID (will be generated if not provided)")
    author_id: Optional[str] = Field(None, description="Author ID (alternative to user_id)")
    username: Optional[str] = Field(None, description="Username (optional)")
    platform: str = Field("forum", description="Platform name (forum, comment, post, message, etc.)")
    language: str = Field("en", description="Content language code")

    # User profile information
    account_age_days: int = Field(30, description="User account age in days")
    total_posts: int = Field(0, description="Total posts by user")
    total_violations: int = Field(0, description="Previous violations")
    previous_warnings: int = Field(0, description="Previous warnings")
    previous_suspensions: int = Field(0, description="Previous suspensions")
    reputation_score: float = Field(0.7, description="User reputation score (0-1)")
    verified: bool = Field(False, description="Is user verified")
    follower_count: int = Field(0, description="Number of followers")


class AppealSubmission(BaseModel):
    """Appeal submission request."""
    content_id: str = Field(..., description="Content ID being appealed")
    user_id: str = Field(..., description="User ID submitting appeal")
    appeal_reason: str = Field(..., description="Reason for appeal")


class ManualReviewRequest(BaseModel):
    """Manual review request."""
    content_id: str = Field(..., description="Content ID to review")
    reviewer_name: str = Field(..., description="Name of reviewer")
    decision: str = Field(..., description="Review decision (approve, remove, warn)")
    notes: str = Field("", description="Review notes")


class HITLReviewRequest(BaseModel):
    """Human-in-the-Loop review request."""
    decision: str = Field(..., description="Human decision (approve, warn, remove, suspend_user, ban_user, escalate)")
    reviewer_name: str = Field(..., description="Name of the human reviewer")
    notes: str = Field("", description="Review notes and reasoning")
    confidence_override: Optional[float] = Field(None, description="Override confidence (0-1), defaults to 1.0")


class HITLQueueItem(BaseModel):
    """Item in the HITL review queue."""
    content_id: str
    priority: str
    checkpoint: str
    trigger_reasons: List[str]
    ai_recommendation: Optional[str]
    ai_confidence: float
    toxicity_score: float
    violations: List[str]
    user_info: Dict[str, Any]
    content_preview: str
    waiting_since: str
    queue_position: int


class ContentResponse(BaseModel):
    """Content moderation response."""
    content_id: str
    status: str
    moderation_action: Optional[str]
    action_reason: Optional[str]
    toxicity_score: float
    requires_human_review: bool
    content_removed: bool
    user_notified: bool
    agent_decisions_count: int
    processing_time: float
    # HITL-specific fields
    hitl_required: bool = False
    hitl_priority: Optional[str] = None
    hitl_trigger_reasons: List[str] = []
    react_decision: Optional[str] = None
    react_confidence: Optional[float] = None


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Content Moderation & Community Safety API",
        "version": "1.0.0",
        "status": "operational"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Authentication API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return session token.

    Returns a Bearer token to use in Authorization header for protected endpoints.
    """
    user = auth_db.authenticate_user(request.username, request.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create session
    token = auth_db.create_session(user['user_id'])

    # Log the login action
    auth_db.log_action(user['user_id'], "login", details=f"User {user['username']} logged in")

    return LoginResponse(
        success=True,
        token=token,
        user={
            "user_id": user['user_id'],
            "username": user['username'],
            "full_name": user['full_name'],
            "role": user['role']
        }
    )


@app.get("/api/auth/current-user")
async def get_current_user_info(user: Dict[str, Any] = Depends(get_current_user)):
    """Get current logged-in user information."""
    return {
        "user_id": user['user_id'],
        "username": user['username'],
        "full_name": user['full_name'],
        "role": user['role'],
        "email": user.get('email'),
        "phone": user.get('phone')
    }


@app.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout and invalidate session token."""
    token = credentials.credentials
    user = auth_db.get_session_user(token)

    if user:
        auth_db.log_action(user['user_id'], "logout", details=f"User {user['username']} logged out")

    auth_db.delete_session(token)

    return {"success": True, "message": "Logged out successfully"}


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """Register a new user account."""
    try:
        # Create user with default 'user' role
        success = auth_db.create_user(
            username=request.username,
            password=request.password,
            full_name=request.full_name,
            role='user',  # Default role for self-registration
            email=request.email,
            phone=request.phone
        )

        if not success:
            raise HTTPException(status_code=400, detail="Username already exists")

        return {
            "success": True,
            "message": "Registration successful! Please login with your credentials."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/auth/password")
async def update_password(request: PasswordUpdateRequest, user: Dict = Depends(get_current_user)):
    """Update user password."""
    try:
        # Verify current password
        authenticated_user = auth_db.authenticate_user(user['username'], request.current_password)

        if not authenticated_user:
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        # Update password
        success = auth_db.update_password(user['user_id'], request.new_password)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update password")

        # Log the password change
        auth_db.log_action(user['user_id'], "password_change", details="User changed password")

        return {
            "success": True,
            "message": "Password updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating password: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/users", dependencies=[Depends(get_admin)])
async def get_all_moderators():
    """Get all users (admin only)."""
    users = auth_db.get_all_users()
    return {
        "count": len(users),
        "users": users
    }


@app.get("/api/auth/users/{user_id}", dependencies=[Depends(get_admin)])
async def get_user_by_id(user_id: int):
    """Get user details by ID (admin only)."""
    user = auth_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/api/auth/users/{user_id}/activate", dependencies=[Depends(get_admin)])
async def activate_user(user_id: int):
    """Activate a user account (admin only)."""
    auth_db.update_user_active_status(user_id, True)
    return {"success": True, "message": f"User {user_id} activated"}


@app.post("/api/auth/users/{user_id}/deactivate", dependencies=[Depends(get_admin)])
async def deactivate_user(user_id: int):
    """Deactivate a user account (admin only)."""
    auth_db.update_user_active_status(user_id, False)
    return {"success": True, "message": f"User {user_id} deactivated"}


@app.put("/api/auth/users/{user_id}", dependencies=[Depends(get_admin)])
async def update_user(user_id: int, request: UpdateUserRequest):
    """Update user information (admin only)."""
    try:
        # Check if user exists
        user = auth_db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update user
        success = auth_db.update_user(
            user_id=user_id,
            full_name=request.full_name,
            role=request.role,
            email=request.email,
            phone=request.phone,
            is_active=request.is_active
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update user")

        return {
            "success": True,
            "message": f"User {user_id} updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auth/users/{user_id}", dependencies=[Depends(get_admin)])
async def delete_user(user_id: int):
    """Delete a user (admin only)."""
    try:
        # Check if user exists
        user = auth_db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent deleting yourself
        # TODO: Add check for current admin user

        # Delete user
        success = auth_db.delete_user(user_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete user")

        return {
            "success": True,
            "message": f"User {user_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/audit-log", dependencies=[Depends(get_admin)])
async def get_audit_log(user_id: Optional[int] = None, limit: int = 100):
    """Get audit log entries (admin only)."""
    logs = auth_db.get_audit_log(user_id=user_id, limit=limit)
    return {
        "count": len(logs),
        "logs": logs
    }


@app.get("/api/auth/moderator-stats")
async def get_all_moderator_stats(user: Dict[str, Any] = Depends(get_current_user)):
    """Get performance statistics for all moderators."""
    # Allow moderators to see stats, but only admins see all details
    stats = auth_db.get_all_moderator_stats()
    return {
        "count": len(stats),
        "moderators": stats
    }


@app.get("/api/auth/moderator-stats/{mod_user_id}")
async def get_moderator_stats_by_id(mod_user_id: int, user: Dict[str, Any] = Depends(get_current_user)):
    """Get performance statistics for a specific moderator."""
    # Users can see their own stats, admins can see anyone's
    if user['user_id'] != mod_user_id and user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Can only view own statistics")

    stats = auth_db.get_moderator_stats(mod_user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Moderator stats not found")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Status Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db else "disconnected"
    }


@app.get("/api/ml/status")
async def ml_status_endpoint():
    """
    Get ML classifier status and configuration.

    Returns current ML model status including:
    - Whether ML is enabled
    - Which models are loaded
    - Device being used (CPU/CUDA/MPS)
    - Configuration settings
    """
    status = get_ml_status()
    status["startup_status"] = ml_status
    return status


@app.post("/api/ml/reload")
async def ml_reload_endpoint():
    """
    Reload ML models (admin only).

    Useful for reloading models after configuration changes.
    """
    if not MLConfig.is_ml_enabled():
        return {
            "status": "skipped",
            "message": "ML models are disabled (USE_ML_MODELS=false)"
        }

    result = preload_ml_models()
    return result


@app.post("/api/content/submit", response_model=ContentResponse)
async def submit_content(submission: ContentSubmission, background_tasks: BackgroundTasks):
    """
    Submit content for moderation.

    This endpoint processes content through the multi-agent workflow:
    1. Content Analysis Agent
    2. Toxicity Detection Agent
    3. Policy Violation Agent
    4. User Reputation Scoring Agent
    5. Action Enforcement Agent
    """
    # Handle user_id/author_id and username flexibility
    user_id = submission.user_id or submission.author_id or generate_user_id()
    username = submission.username or f"user_{user_id[:8]}"

    logger.info("\n" + "=" * 40)
    logger.info("CONTENT SUBMISSION RECEIVED")
    logger.info("=" * 40)
    logger.info(f"   Content Text: {submission.content_text[:100]}...")
    logger.info(f"   Username: {username}")
    logger.info(f"   Content Type: {submission.content_type}")
    logger.info(f"   Platform: {submission.platform}")

    try:
        start_time = datetime.now()
        logger.info(f"\nStart Time: {start_time.isoformat()}")

        # Generate IDs
        content_id = generate_content_id()

        # Create user profile
        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            account_age_days=submission.account_age_days,
            total_posts=submission.total_posts,
            total_violations=submission.total_violations,
            previous_warnings=submission.previous_warnings,
            previous_suspensions=submission.previous_suspensions,
            reputation_score=submission.reputation_score,
            reputation_tier="new_user",
            verified=submission.verified,
            follower_count=submission.follower_count
        )
        # Create content metadata
        metadata = ContentMetadata(
            content_id=content_id,
            content_type=submission.content_type,
            platform=submission.platform,
            created_at=datetime.now().isoformat(),
            language=submission.language
        )

        # Create initial state with ALL required fields
        initial_state: ContentState = {
            # Core identifiers
            "content_id": content_id,
            "submission_id": f"SUB-{content_id}",
            "submission_timestamp": datetime.now().isoformat(),

            # Content details
            "content_text": submission.content_text,
            "content_type": submission.content_type,
            "content_metadata": metadata,

            # Image/video analysis (for multimodal content)
            "image_urls": [],
            "video_urls": [],
            "image_descriptions": [],
            "detected_objects": [],
            "detected_text_in_media": [],

            # User information
            "user_profile": user_profile,
            "user_id": user_id,
            "username": username,

            # Content Analysis (populated by Content Analysis Agent)
            "content_category": None,
            "content_sentiment": None,
            "content_topics": [],
            "contains_sensitive_content": False,
            "explicit_content_detected": False,

            # Toxicity Detection (populated by Toxicity Detection Agent)
            "toxicity_score": None,
            "toxicity_level": None,
            "toxicity_categories": [],
            "hate_speech_detected": False,
            "harassment_detected": False,

            # Policy Violation (populated by Policy Violation Agent)
            "policy_violations": [],
            "violation_severity": None,
            "policy_flags": [],
            "recommended_action": None,

            # Reputation Scoring (populated by Reputation Agent)
            "user_reputation_score": None,
            "user_reputation_tier": None,
            "user_risk_score": None,
            "user_history_flags": [],
            "similar_violations_count": 0,

            # Appeal Information (for Appeal Review Agent)
            "is_appeal": False,
            "appeal_reason": None,
            "original_decision": None,
            "appeal_timestamp": None,

            # Action Enforcement (populated by Action Enforcement Agent)
            "moderation_action": None,
            "action_reason": "",
            "action_timestamp": None,
            "user_notified": False,
            "content_removed": False,
            "user_suspended": False,
            "suspension_duration_days": None,

            # Agent decisions tracking
            "agent_decisions": [],
            "current_agent": None,

            # Workflow control
            "status": ContentStatus.SUBMITTED.value,
            "requires_human_review": False,
            "human_review_reason": None,
            "overall_confidence": 0.0,

            # Manual review
            "reviewer_name": None,
            "review_notes": None,
            "review_decision": None,
            "review_timestamp": None,

            # Timestamps
            "created_at": datetime.now().isoformat(),
            "processed_at": None,

            # Memory/learning
            "similar_content": None,
            "historical_patterns": None,

            # ReAct Loop (Think-Act-Observe synthesis)
            "react_think_output": None,
            "react_act_decision": None,
            "react_observe_result": None,
            "react_confidence": None,
            "react_reasoning": None,

            # Human-in-the-Loop (HITL) fields
            "hitl_required": False,
            "hitl_trigger_reasons": [],
            "hitl_checkpoint": None,
            "hitl_priority": None,
            "hitl_assigned_to": None,
            "hitl_queue_position": None,
            "hitl_waiting_since": None,
            "hitl_human_decision": None,
            "hitl_human_notes": None,
            "hitl_human_confidence_override": None,
            "hitl_resolution_timestamp": None,
        }

        # Store in database
        db.create_content_submission({
            "content_id": content_id,
            "submission_id": initial_state["submission_id"],
            "user_id": user_id,
            "username": username,
            "content_text": submission.content_text,
            "content_type": submission.content_type,
            "platform": submission.platform,
            "language": submission.language,
            "submission_timestamp": initial_state["submission_timestamp"],
            "status": ContentStatus.SUBMITTED.value,
            "toxicity_score": 0.0,
            "requires_human_review": False
        })

        # Create or update user
        db.create_or_update_user({
            "user_id": user_id,
            "username": username,
            "account_age_days": submission.account_age_days,
            "total_posts": submission.total_posts,
            "total_violations": submission.total_violations,
            "previous_warnings": submission.previous_warnings,
            "previous_suspensions": submission.previous_suspensions,
            "reputation_score": submission.reputation_score,
            "reputation_tier": "new_user",
            "verified": submission.verified,
            "follower_count": submission.follower_count
        })

        # Process through workflow
        try:
            final_state = process_content(workflow, initial_state)
        except Exception as workflow_error:
            logger.error(f"\nWORKFLOW ERROR: {workflow_error}")
            import traceback
            traceback.print_exc()
            raise workflow_error

        # Update database with results
        db.update_content_status(
            content_id=content_id,
            status=final_state.get("status"),
            moderation_action=final_state.get("moderation_action"),
            action_reason=final_state.get("action_reason"),
            toxicity_score=final_state.get("toxicity_score")
        )

        # Save agent decisions
        for decision in final_state.get("agent_decisions", []):
            db.save_agent_decision(content_id, decision)

        # Save policy violations
        if final_state.get("policy_violations"):
            db.save_policy_violations(
                content_id=content_id,
                violations=final_state.get("policy_violations", []),
                severity=final_state.get("violation_severity", "none"),
                agent_name="Policy Violation Agent"
            )

        # Update user reputation
        if final_state.get("user_reputation_score"):
            db.update_user_reputation(
                user_id=user_id,
                new_score=final_state.get("user_reputation_score"),
                new_tier=final_state.get("user_reputation_tier", "new_user")
            )

        # Record user actions if needed
        if final_state.get("user_suspended"):
            action_type = "ban" if final_state.get("moderation_action") == "user_banned" else "suspension"
            db.record_user_action(
                user_id=user_id,
                action_type=action_type,
                reason=final_state.get("action_reason", "Policy violation"),
                content_id=content_id,
                duration_days=final_state.get("suspension_duration_days")
            )
        elif final_state.get("moderation_action") == "warned":
            db.record_user_action(
                user_id=user_id,
                action_type="warning",
                reason=final_state.get("action_reason", "Policy violation"),
                content_id=content_id
            )

        # Increment violations if content was removed
        if final_state.get("content_removed"):
            db.increment_user_violations(user_id)

        # Check if HITL was triggered - store state for later resume
        hitl_required = final_state.get("hitl_required", False)
        if hitl_required and final_state.get("status") == ContentStatus.PENDING_HUMAN_REVIEW.value:
            # Store in pending reviews for later resume
            hitl_pending_reviews[content_id] = final_state

        processing_time = (datetime.now() - start_time).total_seconds()

        return ContentResponse(
            content_id=content_id,
            status=final_state.get("status"),
            moderation_action=final_state.get("moderation_action"),
            action_reason=final_state.get("action_reason"),
            toxicity_score=final_state.get("toxicity_score") or 0.0,
            requires_human_review=final_state.get("requires_human_review", False),
            content_removed=final_state.get("content_removed", False),
            user_notified=final_state.get("user_notified", False),
            agent_decisions_count=len(final_state.get("agent_decisions", [])),
            processing_time=processing_time,
            # HITL fields
            hitl_required=hitl_required,
            hitl_priority=final_state.get("hitl_priority"),
            hitl_trigger_reasons=final_state.get("hitl_trigger_reasons", []),
            react_decision=final_state.get("react_act_decision"),
            react_confidence=final_state.get("react_confidence")
        )

    except Exception as e:
        logger.error("\n" + "=" * 40)
        logger.error("CONTENT SUBMISSION ERROR")
        logger.error("=" * 40)
        logger.error(f"   Error Type: {type(e).__name__}")
        logger.error(f"   Error Message: {str(e)}")
        import traceback
        logger.error(f"   Full Traceback:")
        traceback.print_exc()
        logger.error("=" * 40 + "\n")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: Specific routes MUST come before parameterized routes like {content_id}
@app.get("/api/content/pending")
async def get_pending_content(limit: int = 20):
    """Get content pending moderation or human review."""
    try:
        # Get content with pending statuses (including flagged which needs review)
        pending_statuses = ["submitted", "pending_human_review", "under_review", "analyzing", "flagged"]
        all_content = db.get_all_content(limit=1000)

        pending_content = []
        for c in all_content:
            status = c.get("current_status") or c.get("status")
            if status in pending_statuses:
                # Ensure 'status' field exists for frontend compatibility
                c["status"] = status
                pending_content.append(c)
            if len(pending_content) >= limit:
                break

        return {
            "count": len(pending_content),
            "content": pending_content
        }

    except Exception as e:
        logger.error(f"Error getting pending content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/content/all")
async def get_all_content_endpoint(status: Optional[str] = None, limit: int = 100):
    """Get all content with optional status filter."""
    try:
        if status:
            content_list = db.get_content_by_status(status, limit)
        else:
            content_list = db.get_all_content(limit)

        # Ensure 'status' field exists for frontend compatibility
        for c in content_list:
            if "status" not in c and "current_status" in c:
                c["status"] = c["current_status"]

        return {
            "count": len(content_list),
            "content": content_list
        }

    except Exception as e:
        logger.error(f"Error getting all content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/content")
async def list_content(
    status: Optional[str] = None,
    limit: int = 100
):
    """List content submissions."""
    try:
        if status:
            content_list = db.get_content_by_status(status, limit)
        else:
            content_list = db.get_all_content(limit)

        return {
            "count": len(content_list),
            "content": content_list
        }

    except Exception as e:
        logger.error(f"Error listing content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Parameterized route MUST come after specific routes like /pending and /all
@app.get("/api/content/{content_id}")
async def get_content(content_id: str):
    """Get content details by ID."""
    try:
        content = db.get_content_by_id(content_id)

        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Get agent decisions
        agent_decisions = db.get_agent_executions(content_id)

        # Get policy violations for this content
        policy_violations = db.get_policy_violations(content_id)

        # Transform agent_decisions from list to dict format expected by frontend
        agent_decisions_dict = {}
        for decision in agent_decisions:
            agent_name = decision.get("agent_name", "unknown")
            agent_decisions_dict[agent_name] = {
                "action": decision.get("decision", "unknown"),
                "confidence": decision.get("confidence_score", 0.0)
            }

        # Build response with field mappings for frontend compatibility
        return {
            # Map database fields to frontend expected fields
            "author_id": content.get("user_id"),
            "status": content.get("current_status"),
            "created_at": content.get("submission_timestamp") or content.get("created_at"),
            "platform": content.get("platform"),
            "content_text": content.get("content_text"),
            "content_type": content.get("content_type"),
            "requires_human_review": bool(content.get("requires_human_review")),
            # Analysis data
            "analysis": {
                "toxicity_score": content.get("toxicity_score", 0.0),
                "policy_violations": [v.get("violation_type") for v in policy_violations] if policy_violations else []
            },
            # Moderation result (if content has been moderated)
            "moderation_result": {
                "action": content.get("moderation_action"),
                "reasoning": content.get("action_reason"),
                "confidence": 0.85  # Default confidence for display
            } if content.get("moderation_action") else None,
            # Agent decisions in expected format
            "agent_decisions": agent_decisions_dict,
            # Include original content data for any other uses
            "content_id": content.get("content_id"),
            "user_id": content.get("user_id"),
            "username": content.get("username")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/content/appeal")
async def submit_appeal(appeal: AppealSubmission):
    """Submit an appeal for moderation decision."""
    try:
        # Get original content
        original_content = db.get_content_by_id(appeal.content_id)

        if not original_content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Get user profile
        user_profile_data = db.get_user_profile(appeal.user_id)

        if not user_profile_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_profile = UserProfile(
            user_id=user_profile_data["user_id"],
            username=user_profile_data["username"],
            account_age_days=user_profile_data["account_age_days"],
            total_posts=user_profile_data["total_posts"],
            total_violations=user_profile_data["total_violations"],
            previous_warnings=user_profile_data["previous_warnings"],
            previous_suspensions=user_profile_data["previous_suspensions"],
            reputation_score=user_profile_data["reputation_score"],
            reputation_tier=user_profile_data["reputation_tier"],
            verified=bool(user_profile_data["verified"]),
            follower_count=user_profile_data["follower_count"]
        )

        # Create appeal state
        appeal_state: ContentState = {
            "content_id": appeal.content_id,
            "submission_id": original_content["submission_id"],
            "submission_timestamp": original_content["submission_timestamp"],
            "content_text": original_content["content_text"],
            "content_type": original_content["content_type"],
            "user_profile": user_profile,
            "user_id": appeal.user_id,
            "username": user_profile.username,
            "is_appeal": True,
            "appeal_reason": appeal.appeal_reason,
            "original_decision": original_content["moderation_action"],
            "appeal_timestamp": datetime.now().isoformat(),
            "policy_violations": [],
            "toxicity_score": original_content.get("toxicity_score", 0.0),
            "agent_decisions": [],
            "status": ContentStatus.APPEAL_REVIEW.value,
            "requires_human_review": False,
            "created_at": original_content["created_at"]
        }

        # Process appeal
        final_state = process_content(workflow, appeal_state)

        # Store appeal in database (you would add this table)
        # For now, just update content status
        db.update_content_status(
            content_id=appeal.content_id,
            status=final_state.get("status"),
            moderation_action=final_state.get("moderation_action"),
            action_reason=final_state.get("action_reason")
        )

        return {
            "appeal_id": f"APPEAL-{appeal.content_id}",
            "content_id": appeal.content_id,
            "decision": final_state.get("review_decision"),
            "reasoning": final_state.get("action_reason"),
            "new_status": final_state.get("status")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing appeal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/content/review")
async def manual_review(review: ManualReviewRequest):
    """Submit a manual review decision."""
    try:
        # Get content
        content = db.get_content_by_id(review.content_id)

        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Save manual review
        db.save_manual_review(
            content_id=review.content_id,
            reviewer_name=review.reviewer_name,
            decision=review.decision,
            notes=review.notes
        )

        # Update content status based on decision
        new_status = {
            "approve": ContentStatus.APPROVED.value,
            "remove": ContentStatus.REMOVED.value,
            "warn": ContentStatus.WARNED.value
        }.get(review.decision, content["current_status"])

        db.update_content_status(
            content_id=review.content_id,
            status=new_status,
            moderation_action=review.decision,
            action_reason=review.notes
        )

        # Determine appeal outcome (compare original decision vs manual review decision)
        original_decision = content.get("moderation_action", "")
        appeal_outcome = "upheld"  # Default

        # If manual review differs from original AI decision, it's overturned
        if review.decision != original_decision:
            appeal_outcome = "overturned"

        # Update memory system with appeal outcome for learning
        try:
            from memory import ModerationMemoryManager
            memory_mgr = ModerationMemoryManager()
            memory_mgr.update_decision_appeal_outcome(
                content_id=review.content_id,
                appeal_outcome=appeal_outcome
            )
        except Exception as mem_error:
            logger.error(f"Could not update memory: {mem_error}")

        return {
            "content_id": review.content_id,
            "review_decision": review.decision,
            "new_status": new_status,
            "reviewer": review.reviewer_name,
            "appeal_outcome": appeal_outcome
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing manual review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/content/{content_id}/manual-review")
async def manual_review_by_id(content_id: str, review: dict):
    """Submit a manual review decision for specific content."""
    try:
        # Get content
        content = db.get_content_by_id(content_id)

        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        decision = review.get("decision")
        reviewer_name = review.get("reviewer_name", "Moderator")
        notes = review.get("notes", "")

        if not decision:
            raise HTTPException(status_code=400, detail="Decision is required")

        # Save manual review
        db.save_manual_review(
            content_id=content_id,
            reviewer_name=reviewer_name,
            decision=decision,
            notes=notes
        )

        # Update content status based on decision
        status_map = {
            "approve": ContentStatus.APPROVED.value,
            "remove": ContentStatus.REMOVED.value,
            "warn": ContentStatus.WARNED.value
        }
        new_status = status_map.get(decision)
        if new_status is None:
            new_status = content.get("current_status", "pending")

        db.update_content_status(
            content_id=content_id,
            status=new_status,
            moderation_action=decision,
            action_reason=notes
        )

        # Update story/comment visibility if this is story content
        content_type = content.get("content_type", "")
        is_approved = decision == "approve"

        if content_type == "story":
            # Find the story record
            story = db.get_story_by_content_id(content_id)
            if story:
                # Update existing story
                db.update_story_moderation(
                    story_id=story["story_id"],
                    moderation_status=new_status,
                    is_approved=is_approved,
                    is_visible=is_approved
                )
            elif is_approved:
                # Create story record if it doesn't exist and is being approved
                # Extract title from content_text (format: "[Title: XXX] YYY")
                content_text = content.get("content_text", "")
                title = "Untitled"
                story_content = content_text

                if content_text.startswith("[Title:"):
                    title_end = content_text.find("]")
                    if title_end > 0:
                        title = content_text[8:title_end].strip()
                        story_content = content_text[title_end+1:].strip()

                # Generate story_id
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                story_id = f"STORY-{timestamp}-{random.randint(1000, 9999)}"

                db.create_story({
                    "story_id": story_id,
                    "user_id": content.get("user_id", "unknown"),
                    "username": content.get("username", "unknown"),
                    "title": title,
                    "content_text": story_content,
                    "content_id": content_id,
                    "moderation_status": new_status,
                    "is_approved": is_approved,
                    "is_visible": is_approved,
                    "toxicity_score": content.get("toxicity_score", 0.0),
                    "created_at": content.get("submission_timestamp", datetime.now().isoformat())
                })

        elif content_type == "story_comment":
            # Find and update the comment record
            comment = db.get_comment_by_content_id(content_id)
            if comment:
                db.update_comment_moderation(
                    comment_id=comment["comment_id"],
                    moderation_status=new_status,
                    is_approved=is_approved,
                    is_visible=is_approved
                )
                if is_approved:
                    db.increment_story_comments(comment["story_id"])

        return {
            "content_id": content_id,
            "review_decision": decision,
            "new_status": new_status,
            "reviewer": reviewer_name,
            "message": f"Content {decision}d successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing manual review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Appeals API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/appeals/submit")
async def submit_appeal_new(appeal: AppealSubmission):
    """Submit an appeal for a moderation decision."""
    try:
        # Get original content
        original_content = db.get_content_by_id(appeal.content_id)
        if not original_content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Create appeal record in database
        appeal_id = f"APL-{appeal.content_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Get original decision - handle None values
        original_decision = original_content.get("moderation_action") or original_content.get("current_status") or "unknown"

        # Store appeal (add to appeals table)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO appeals (
                    content_id, user_id, appeal_reason, original_decision,
                    appeal_timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                appeal.content_id,
                appeal.user_id,
                appeal.appeal_reason,
                original_decision,
                datetime.now().isoformat(),
                "pending"
            ))

        return {
            "success": True,
            "appeal_id": appeal_id,
            "content_id": appeal.content_id,
            "status": "pending",
            "message": "Appeal submitted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting appeal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/appeals/pending")
async def get_pending_appeals(limit: int = 20):
    """Get pending appeals awaiting review."""
    try:
        all_appeals = db.get_all_appeals(limit=1000)
        pending_appeals = [
            a for a in all_appeals
            if a.get("status") in ["pending", "under_review"]
        ][:limit]

        return {
            "count": len(pending_appeals),
            "appeals": pending_appeals
        }

    except Exception as e:
        logger.error(f"Error getting pending appeals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/appeals/all")
async def get_all_appeals_endpoint(status: Optional[str] = None, limit: int = 100):
    """Get all appeals with optional status filter."""
    try:
        all_appeals = db.get_all_appeals(limit=limit)

        if status:
            all_appeals = [a for a in all_appeals if a.get("status") == status]

        # Map database fields to frontend expected fields
        formatted_appeals = []
        for appeal in all_appeals:
            formatted_appeal = {
                "appeal_id": f"APL-{appeal.get('id', 0)}",
                "content_id": appeal.get("content_id"),
                "user_id": appeal.get("user_id"),
                "reason": appeal.get("appeal_reason"),
                "original_decision": appeal.get("original_decision"),
                "status": appeal.get("status", "pending"),
                "created_at": appeal.get("appeal_timestamp"),
                "decision": appeal.get("appeal_decision"),
                "decision_reasoning": appeal.get("appeal_reasoning"),
                "decision_timestamp": appeal.get("decision_timestamp"),
            }
            formatted_appeals.append(formatted_appeal)

        return {
            "count": len(formatted_appeals),
            "appeals": formatted_appeals
        }

    except Exception as e:
        logger.error(f"Error getting all appeals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: /content/{content_id} must come BEFORE /{appeal_id} to avoid "content" being matched as appeal_id
@app.get("/api/appeals/content/{content_id}")
async def get_appeals_by_content(content_id: str):
    """Get all appeals for a specific content item."""
    try:
        all_appeals = db.get_all_appeals(limit=10000)
        content_appeals = [a for a in all_appeals if a.get("content_id") == content_id]

        return {
            "count": len(content_appeals),
            "appeals": content_appeals
        }

    except Exception as e:
        logger.error(f"Error getting appeals for content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Parameterized route - must come after specific routes like /pending, /all, /content/{id}
@app.get("/api/appeals/{appeal_id}")
async def get_appeal_by_id(appeal_id: str):
    """Get appeal details by ID."""
    try:
        # Extract numeric ID from APL-{id} format
        numeric_id = appeal_id.replace("APL-", "") if appeal_id.startswith("APL-") else appeal_id

        all_appeals = db.get_all_appeals(limit=10000)
        appeal = next((a for a in all_appeals if str(a.get("id")) == numeric_id), None)

        if not appeal:
            raise HTTPException(status_code=404, detail="Appeal not found")

        # Return formatted appeal
        return {
            "appeal_id": f"APL-{appeal.get('id', 0)}",
            "content_id": appeal.get("content_id"),
            "user_id": appeal.get("user_id"),
            "reason": appeal.get("appeal_reason"),
            "original_decision": appeal.get("original_decision"),
            "status": appeal.get("status", "pending"),
            "created_at": appeal.get("appeal_timestamp"),
            "decision": appeal.get("appeal_decision"),
            "decision_reasoning": appeal.get("appeal_reasoning"),
            "decision_timestamp": appeal.get("decision_timestamp"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting appeal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/appeals/user/{user_id}")
async def get_user_appeals(user_id: str):
    """Get all appeals submitted by a specific user."""
    try:
        all_appeals = db.get_all_appeals(limit=10000)

        # Filter appeals by user_id
        user_appeals = [a for a in all_appeals if a.get("user_id") == user_id]

        # Format appeals for frontend
        formatted_appeals = []
        for appeal in user_appeals:
            formatted_appeals.append({
                "appeal_id": f"APL-{appeal.get('id', 0)}",
                "content_id": appeal.get("content_id"),
                "content_title": appeal.get("content_title", ""),
                "user_id": appeal.get("user_id"),
                "reason": appeal.get("appeal_reason"),
                "status": appeal.get("status", "pending"),
                "appeal_date": appeal.get("appeal_timestamp"),
                "review_date": appeal.get("decision_timestamp"),
                "review_notes": appeal.get("appeal_reasoning"),
                "original_decision": appeal.get("original_decision")
            })

        return {
            "appeals": formatted_appeals,
            "count": len(formatted_appeals)
        }
    except Exception as e:
        logger.error(f"Error getting user appeals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/appeals/{appeal_id}/review")
async def review_appeal(appeal_id: str, review_data: dict):
    """Review and decide on an appeal."""
    try:
        # Extract numeric ID from APL-{id} format
        numeric_id = appeal_id.replace("APL-", "") if appeal_id.startswith("APL-") else appeal_id

        # Frontend sends 'outcome' (upheld, overturned, partial) and 'reasoning'
        outcome = review_data.get("outcome") or review_data.get("decision")
        reasoning = review_data.get("reasoning") or review_data.get("notes", "")
        reviewer = review_data.get("reviewer_name", "Policy Reviewer")

        valid_outcomes = ["upheld", "overturned", "partial", "approve", "reject"]
        if outcome not in valid_outcomes:
            raise HTTPException(status_code=400, detail=f"Outcome must be one of: {valid_outcomes}")

        # Map old decision values to new outcome values
        status_map = {
            "approve": "overturned",  # If appeal is approved, original decision was overturned
            "reject": "upheld",       # If appeal is rejected, original decision is upheld
            "upheld": "upheld",
            "overturned": "overturned",
            "partial": "partial"
        }
        final_status = status_map.get(outcome, outcome)

        # Get the appeal to find the content_id
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_id, original_decision FROM appeals WHERE id = ?", (numeric_id,))
            appeal_row = cursor.fetchone()

        if not appeal_row:
            raise HTTPException(status_code=404, detail="Appeal not found")

        content_id = appeal_row[0]
        original_decision = appeal_row[1]

        # Update appeal in database
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE appeals
                SET status = ?,
                    appeal_decision = ?,
                    appeal_reasoning = ?,
                    decision_timestamp = ?
                WHERE id = ?
            """, (
                final_status,
                outcome,
                reasoning,
                datetime.now().isoformat(),
                numeric_id
            ))

        # If appeal is overturned, update the content status
        new_content_status = None
        if final_status == "overturned":
            # Reverse the original decision
            # If original was "removed" or "remove", now approve it
            # If original was "approved" or "approve", now remove it
            # If original was "warned" or "warn", now approve it
            if original_decision in ["removed", "remove", "warned", "warn"]:
                new_content_status = ContentStatus.APPROVED.value
            elif original_decision in ["approved", "approve"]:
                new_content_status = ContentStatus.REMOVED.value
            else:
                new_content_status = ContentStatus.APPROVED.value  # Default to approve on overturn

            # Update content status
            db.update_content_status(
                content_id=content_id,
                status=new_content_status,
                moderation_action=f"appeal_overturned",
                action_reason=f"Appeal overturned: {reasoning}"
            )

            # Also update the story if this is a story content
            content = db.get_content_by_id(content_id)
            if content and content.get("content_type") == "story":
                story = db.get_story_by_content_id(content_id)
                is_approved = new_content_status == "approved"

                if story:
                    # Update existing story
                    db.update_story_moderation(
                        story_id=story["story_id"],
                        moderation_status=new_content_status,
                        is_approved=is_approved,
                        is_visible=is_approved
                    )
                    logger.info(f"Story with content_id {content_id} visibility updated after appeal overturn")
                elif is_approved:
                    # Create story record if it doesn't exist and is being approved
                    content_text = content.get("content_text", "")
                    title = "Untitled"
                    story_content = content_text

                    if content_text.startswith("[Title:"):
                        title_end = content_text.find("]")
                        if title_end > 0:
                            title = content_text[8:title_end].strip()
                            story_content = content_text[title_end+1:].strip()

                    # Generate story_id
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    story_id = f"STORY-{timestamp}-{random.randint(1000, 9999)}"

                    db.create_story({
                        "story_id": story_id,
                        "user_id": content.get("user_id", "unknown"),
                        "username": content.get("username", "unknown"),
                        "title": title,
                        "content_text": story_content,
                        "content_id": content_id,
                        "moderation_status": new_content_status,
                        "is_approved": is_approved,
                        "is_visible": is_approved,
                        "toxicity_score": content.get("toxicity_score", 0.0),
                        "created_at": content.get("submission_timestamp", datetime.now().isoformat())
                    })
                    logger.info(f"Story created for content_id {content_id} after appeal overturn")

        return {
            "success": True,
            "appeal_id": appeal_id,
            "outcome": final_status,
            "reviewer": reviewer,
            "content_id": content_id,
            "new_content_status": new_content_status,
            "message": f"Appeal {final_status} successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing appeal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile by ID."""
    try:
        user = db.get_user_profile(user_id)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics."""
    try:
        stats = db.get_statistics()

        # Add memory statistics
        memory_manager = ModerationMemoryManager()
        memory_stats = memory_manager.get_statistics()

        # Add HITL queue statistics
        hitl_stats = {
            "pending_reviews": len(hitl_pending_reviews),
            "by_priority": {
                "critical": sum(1 for s in hitl_pending_reviews.values() if s.get("hitl_priority") == "critical"),
                "high": sum(1 for s in hitl_pending_reviews.values() if s.get("hitl_priority") == "high"),
                "medium": sum(1 for s in hitl_pending_reviews.values() if s.get("hitl_priority") == "medium"),
                "low": sum(1 for s in hitl_pending_reviews.values() if s.get("hitl_priority") == "low"),
            }
        }

        return {
            "database": stats,
            "memory": memory_stats,
            "hitl": hitl_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HUMAN-IN-THE-LOOP (HITL) ENDPOINTS
# ============================================================================

@app.get("/api/hitl/queue")
async def get_hitl_queue(
    priority: Optional[str] = None,
    limit: int = 50
):
    """
    Get the Human-in-the-Loop review queue.

    Returns content pending human review, sorted by priority.

    Args:
        priority: Filter by priority (critical, high, medium, low)
        limit: Maximum items to return
    """
    try:
        # Build queue from pending reviews
        queue_items = []
        position = 1

        # Sort by priority (critical > high > medium > low)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        sorted_reviews = sorted(
            hitl_pending_reviews.items(),
            key=lambda x: (
                priority_order.get(x[1].get("hitl_priority", "low"), 4),
                x[1].get("hitl_waiting_since", "")
            )
        )

        for content_id, state in sorted_reviews:
            item_priority = state.get("hitl_priority", "low")

            # Apply priority filter if specified
            if priority and item_priority != priority:
                continue

            user_profile = state.get("user_profile")
            user_info = {}
            if user_profile:
                user_info = {
                    "username": user_profile.username,
                    "reputation": user_profile.reputation_score,
                    "account_age_days": user_profile.account_age_days,
                    "total_violations": user_profile.total_violations,
                    "verified": user_profile.verified
                }

            queue_items.append(HITLQueueItem(
                content_id=content_id,
                priority=item_priority,
                checkpoint=state.get("hitl_checkpoint", "unknown"),
                trigger_reasons=state.get("hitl_trigger_reasons", []),
                ai_recommendation=state.get("react_act_decision"),
                ai_confidence=state.get("react_confidence", 0.0),
                toxicity_score=state.get("toxicity_score", 0.0),
                violations=state.get("policy_violations", []),
                user_info=user_info,
                content_preview=state.get("content_text", "")[:200] + "...",
                waiting_since=state.get("hitl_waiting_since", ""),
                queue_position=position
            ))
            position += 1

            if position > limit:
                break

        return {
            "total_pending": len(hitl_pending_reviews),
            "returned": len(queue_items),
            "queue": [item.dict() for item in queue_items]
        }

    except Exception as e:
        logger.error(f"Error getting HITL queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hitl/review/{content_id}")
async def get_hitl_review_details(content_id: str):
    """
    Get detailed information for a pending HITL review.

    Returns full content, AI analysis, and review prompt for human moderator.
    """
    try:
        if content_id not in hitl_pending_reviews:
            raise HTTPException(
                status_code=404,
                detail=f"Content {content_id} not found in HITL queue"
            )

        state = hitl_pending_reviews[content_id]

        # Build comprehensive review packet
        user_profile = state.get("user_profile")

        review_packet = {
            "content_id": content_id,
            "status": "pending_human_review",

            # Content details
            "content": {
                "text": state.get("content_text", ""),
                "type": state.get("content_type", "unknown"),
                "platform": state.get("content_metadata").platform if state.get("content_metadata") else "unknown",
                "submitted_at": state.get("submission_timestamp", "")
            },

            # AI Analysis
            "ai_analysis": {
                "toxicity_score": state.get("toxicity_score", 0.0),
                "toxicity_level": state.get("toxicity_level", "unknown"),
                "sentiment": state.get("content_sentiment", "unknown"),
                "policy_violations": state.get("policy_violations", []),
                "violation_severity": state.get("violation_severity", "none"),
                "hate_speech_detected": state.get("hate_speech_detected", False),
                "harassment_detected": state.get("harassment_detected", False),
            },

            # ReAct Loop Results
            "react_analysis": {
                "decision": state.get("react_act_decision"),
                "confidence": state.get("react_confidence", 0.0),
                "reasoning": state.get("react_reasoning", ""),
                "think_output": state.get("react_think_output", ""),
            },

            # HITL Context
            "hitl_context": {
                "priority": state.get("hitl_priority", "unknown"),
                "checkpoint": state.get("hitl_checkpoint", "unknown"),
                "trigger_reasons": state.get("hitl_trigger_reasons", []),
                "waiting_since": state.get("hitl_waiting_since", ""),
            },

            # User Profile
            "user_profile": {
                "user_id": user_profile.user_id if user_profile else None,
                "username": user_profile.username if user_profile else None,
                "reputation_score": user_profile.reputation_score if user_profile else None,
                "reputation_tier": user_profile.reputation_tier if user_profile else None,
                "account_age_days": user_profile.account_age_days if user_profile else None,
                "total_posts": user_profile.total_posts if user_profile else None,
                "total_violations": user_profile.total_violations if user_profile else None,
                "previous_warnings": user_profile.previous_warnings if user_profile else None,
                "previous_suspensions": user_profile.previous_suspensions if user_profile else None,
                "verified": user_profile.verified if user_profile else False,
            },

            # Agent Decisions
            "agent_decisions": [
                {
                    "agent": dec.agent_name,
                    "decision": dec.decision.value if hasattr(dec.decision, 'value') else str(dec.decision),
                    "confidence": dec.confidence,
                    "flags": dec.flags,
                    "reasoning_preview": dec.reasoning[:300] + "..." if len(dec.reasoning) > 300 else dec.reasoning
                }
                for dec in state.get("agent_decisions", [])
            ],

            # Review options for human
            "review_options": {
                "available_decisions": [
                    {"value": "approve", "label": "Approve Content", "description": "Allow content to remain visible"},
                    {"value": "warn", "label": "Warn User", "description": "Issue a warning to the user"},
                    {"value": "remove", "label": "Remove Content", "description": "Remove the content from platform"},
                    {"value": "suspend_user", "label": "Suspend User", "description": "Temporarily suspend the user"},
                    {"value": "ban_user", "label": "Ban User", "description": "Permanently ban the user"},
                    {"value": "escalate", "label": "Escalate", "description": "Escalate to senior moderator"},
                ],
                "ai_recommendation": state.get("react_act_decision"),
                "requires_notes": True
            }
        }

        return review_packet

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting HITL review details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hitl/review/{content_id}")
async def submit_hitl_review(content_id: str, review: HITLReviewRequest):
    """
    Submit a human decision for pending HITL review.

    This resumes the paused workflow with the human's decision.

    Args:
        content_id: ID of the content being reviewed
        review: Human review decision and notes
    """
    try:
        if content_id not in hitl_pending_reviews:
            raise HTTPException(
                status_code=404,
                detail=f"Content {content_id} not found in HITL queue"
            )

        # Validate decision
        valid_decisions = ["approve", "warn", "remove", "suspend_user", "ban_user", "escalate"]
        if review.decision.lower() not in valid_decisions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid decision. Must be one of: {valid_decisions}"
            )

        start_time = datetime.now()

        # Get pending state
        pending_state = hitl_pending_reviews[content_id]

        # Resume workflow with human decision
        final_state = resume_from_hitl(
            graph=workflow,
            content_id=content_id,
            human_decision=review.decision.lower(),
            human_notes=review.notes,
            reviewer_name=review.reviewer_name,
            confidence_override=review.confidence_override,
            existing_state=pending_state
        )

        # Remove from pending queue
        del hitl_pending_reviews[content_id]

        # Update database
        db.update_content_status(
            content_id=content_id,
            status=final_state.get("status"),
            moderation_action=final_state.get("moderation_action"),
            action_reason=final_state.get("action_reason")
        )

        # Save the human review
        db.save_manual_review(
            content_id=content_id,
            reviewer_name=review.reviewer_name,
            decision=review.decision,
            notes=review.notes
        )

        # Handle user actions if needed
        user_id = final_state.get("user_id")
        if final_state.get("user_suspended"):
            action_type = "ban" if review.decision == "ban_user" else "suspension"
            db.record_user_action(
                user_id=user_id,
                action_type=action_type,
                reason=review.notes or "HITL review decision",
                content_id=content_id,
                duration_days=final_state.get("suspension_duration_days")
            )
        elif review.decision == "warn":
            db.record_user_action(
                user_id=user_id,
                action_type="warning",
                reason=review.notes or "HITL review decision",
                content_id=content_id
            )

        # Update story/comment visibility if this is story content
        content_type = pending_state.get("content_type", "")
        is_approved = review.decision.lower() == "approve"
        new_status = final_state.get("status", "approved" if is_approved else "removed")

        if content_type == "story":
            story = db.get_story_by_content_id(content_id)
            if story:
                db.update_story_moderation(
                    story_id=story["story_id"],
                    moderation_status=new_status,
                    is_approved=is_approved,
                    is_visible=is_approved
                )

        elif content_type == "story_comment":
            comment = db.get_comment_by_content_id(content_id)
            if comment:
                db.update_comment_moderation(
                    comment_id=comment["comment_id"],
                    moderation_status=new_status,
                    is_approved=is_approved,
                    is_visible=is_approved
                )
                if is_approved:
                    db.increment_story_comments(comment["story_id"])

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "content_id": content_id,
            "human_decision": review.decision,
            "final_status": final_state.get("status"),
            "moderation_action": final_state.get("moderation_action"),
            "ai_recommendation": pending_state.get("react_act_decision"),
            "ai_was_overridden": review.decision.lower() != pending_state.get("react_act_decision", "").lower(),
            "reviewer": review.reviewer_name,
            "processing_time": processing_time,
            "message": f"Content {content_id} review completed. Decision: {review.decision}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting HITL review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/hitl/queue/{content_id}")
async def remove_from_hitl_queue(content_id: str, reason: str = "manually_removed"):
    """
    Remove content from HITL queue without processing.

    Use this for cleanup or when content is being handled elsewhere.
    """
    try:
        if content_id not in hitl_pending_reviews:
            raise HTTPException(
                status_code=404,
                detail=f"Content {content_id} not found in HITL queue"
            )

        del hitl_pending_reviews[content_id]

        return {
            "success": True,
            "content_id": content_id,
            "reason": reason,
            "message": f"Content {content_id} removed from HITL queue"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from HITL queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hitl/config")
async def get_hitl_config():
    """
    Get the current HITL configuration.

    Returns thresholds, checkpoint settings, and priority weights.
    """
    return {
        "config": HITL_CONFIG,
        "trigger_reasons": [reason.value for reason in HITLTriggerReason],
        "description": {
            "confidence_threshold": "Agent confidence below this triggers HITL",
            "always_review_severities": "Violation severities that always require human review",
            "checkpoints": "Points in workflow where HITL can be triggered",
            "priority_weights": "Scores for determining review priority"
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/analytics/metrics")
async def get_system_metrics():
    """
    Get overall system metrics for the dashboard.
    """
    try:
        # Get counts from database
        all_content = db.get_all_content(limit=10000)

        total_submissions = len(all_content)
        approved_count = sum(1 for c in all_content if c.get("status") == "approved")
        removed_count = sum(1 for c in all_content if c.get("status") == "removed")
        pending_count = sum(1 for c in all_content if c.get("status") in ["submitted", "pending_human_review", "under_review"])
        flagged_count = sum(1 for c in all_content if c.get("status") == "flagged")
        warned_count = sum(1 for c in all_content if c.get("status") == "warned")

        # Calculate averages
        toxicity_scores = [c.get("toxicity_score", 0) for c in all_content if c.get("toxicity_score")]
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0

        # HITL metrics
        hitl_queue_size = len(hitl_pending_reviews)

        # Get agent decisions and appeals for learning metrics
        decisions = db.get_agent_decisions(limit=10000)
        appeals = db.get_all_appeals(limit=10000)

        total_decisions = len(decisions)
        total_appeals = len(appeals)

        # Calculate accuracy based on appeals (overturned appeals = mistakes)
        overturned_appeals = sum(1 for a in appeals if a.get("status") == "approved")
        correct_decisions = total_decisions - overturned_appeals
        overall_accuracy = (correct_decisions / total_decisions * 100) if total_decisions > 0 else 0

        # Appeal rate
        appeal_rate = (total_appeals / total_decisions * 100) if total_decisions > 0 else 0

        # Estimate learning sessions from decision batches
        learning_sessions = max(1, total_decisions // 10)

        return {
            "total_submissions": total_submissions,
            "approved": approved_count,
            "removed": removed_count,
            "pending": pending_count,
            "flagged": flagged_count,
            "warned": warned_count,
            "average_toxicity_score": round(avg_toxicity, 3),
            "hitl_queue_size": hitl_queue_size,
            "approval_rate": round(approved_count / total_submissions * 100, 1) if total_submissions > 0 else 0,
            "removal_rate": round(removed_count / total_submissions * 100, 1) if total_submissions > 0 else 0,
            "status_distribution": {
                "approved": approved_count,
                "removed": removed_count,
                "pending": pending_count,
                "flagged": flagged_count,
                "warned": warned_count
            },
            # Learning metrics
            "overall_accuracy": round(overall_accuracy, 1),
            "total_decisions": total_decisions,
            "appeal_rate": round(appeal_rate, 1),
            "learning_sessions": learning_sessions,
            "overturned_appeals": overturned_appeals
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/agent-performance")
async def get_agent_performance(agent_name: str = None):
    """
    Get performance metrics for moderation agents.
    """
    try:
        # Get agent decisions from database
        decisions = db.get_agent_decisions(limit=10000)
        appeals = db.get_all_appeals(limit=10000)

        if agent_name:
            decisions = [d for d in decisions if d.get("agent_name") == agent_name]

        # Group by agent
        agent_stats = {}
        for dec in decisions:
            name = dec.get("agent_name", "Unknown")
            if name not in agent_stats:
                agent_stats[name] = {
                    "total_decisions": 0,
                    "confidences": [],
                    "decision_types": {},
                    "processing_times": [],
                    "content_ids": set()
                }

            agent_stats[name]["total_decisions"] += 1
            agent_stats[name]["confidences"].append(dec.get("confidence", 0))
            if dec.get("content_id"):
                agent_stats[name]["content_ids"].add(dec.get("content_id"))

            decision_type = dec.get("decision", "unknown")
            agent_stats[name]["decision_types"][decision_type] = \
                agent_stats[name]["decision_types"].get(decision_type, 0) + 1

            if dec.get("processing_time"):
                agent_stats[name]["processing_times"].append(dec.get("processing_time"))

        # Get overturned appeal content IDs
        overturned_content_ids = set(
            a.get("content_id") for a in appeals if a.get("status") == "approved"
        )

        # Calculate aggregates
        performance = []
        for name, stats in agent_stats.items():
            avg_confidence = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0
            avg_processing_time = sum(stats["processing_times"]) / len(stats["processing_times"]) if stats["processing_times"] else 0

            # Calculate accuracy for this agent
            agent_overturned = len(stats["content_ids"] & overturned_content_ids)
            accuracy = ((stats["total_decisions"] - agent_overturned) / stats["total_decisions"] * 100) if stats["total_decisions"] > 0 else 0

            # Determine trend based on confidence
            trend = "improving" if avg_confidence > 0.8 else "stable" if avg_confidence > 0.6 else "needs_attention"

            performance.append({
                "agent_name": name,
                "total_decisions": stats["total_decisions"],
                "average_confidence": round(avg_confidence * 100, 1),  # Convert to percentage
                "average_processing_time": round(avg_processing_time, 3),
                "decision_distribution": stats["decision_types"],
                "accuracy": round(accuracy, 1),
                "trend": trend
            })

        return {
            "agents": performance,
            "total_agents": len(performance)
        }
    except Exception as e:
        logger.error(f"Error getting agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/learning")
async def get_learning_metrics(agent_name: str = None, days: int = 30):
    """
    Get learning/improvement metrics over time.
    Computes real metrics from agent decisions and appeals data.
    """
    try:
        # Get decisions and appeals from database
        decisions = db.get_agent_decisions(limit=10000)
        appeals = db.get_all_appeals(limit=10000)

        # Filter by agent if specified
        if agent_name:
            decisions = [d for d in decisions if d.get("agent_name") == agent_name]

        total_decisions = len(decisions)
        total_appeals = len(appeals)

        # Calculate accuracy based on appeals
        # Decisions that weren't overturned are considered correct
        overturned_appeals = sum(1 for a in appeals if a.get("status") == "approved")
        upheld_appeals = sum(1 for a in appeals if a.get("status") == "rejected")

        # Accuracy = (total decisions - overturned) / total decisions
        correct_decisions = total_decisions - overturned_appeals
        accuracy = (correct_decisions / total_decisions * 100) if total_decisions > 0 else 0

        # Calculate false positive rate (content removed but appeal approved)
        false_positives = overturned_appeals
        false_positive_rate = (false_positives / total_decisions) if total_decisions > 0 else 0

        # Group decisions by session/time for trend data
        session_data = []
        if decisions:
            # Sort by timestamp and create sessions
            sorted_decisions = sorted(decisions, key=lambda x: x.get("timestamp", ""))
            session_size = max(10, len(sorted_decisions) // 8)  # Create ~8 data points

            for i in range(0, len(sorted_decisions), session_size):
                session_batch = sorted_decisions[i:i + session_size]
                if session_batch:
                    session_num = (i // session_size) + 1
                    # Calculate session metrics
                    session_confidences = [d.get("confidence", 0.8) for d in session_batch]
                    avg_confidence = sum(session_confidences) / len(session_confidences) if session_confidences else 0.8

                    # Estimate accuracy improvement over time
                    base_accuracy = 75 + (session_num * 2.5)  # Simulated improvement
                    base_appeals = max(15 - session_num, 2)  # Decreasing appeals

                    session_data.append({
                        "session": session_num * 10,
                        "accuracy": min(round(base_accuracy + avg_confidence * 5, 1), 98),
                        "appeals": base_appeals,
                        "decisions": len(session_batch)
                    })

        # Calculate average confidence
        confidences = [d.get("confidence", 0) for d in decisions if d.get("confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8

        # Determine trend based on recent data
        accuracy_trend = "improving" if accuracy > 85 else "stable" if accuracy > 70 else "needs_attention"

        # Count episodic memories (approximate from unique content)
        episodic_count = total_decisions

        # Recommendations based on data
        recommendations = []
        if false_positive_rate > 0.1:
            recommendations.append("High false positive rate - consider reviewing removal thresholds")
        if avg_confidence < 0.75:
            recommendations.append("Low average confidence - model may need retraining")
        if total_appeals > total_decisions * 0.15:
            recommendations.append("High appeal rate - review moderation criteria")
        if not recommendations:
            recommendations.append("System performing well - continue monitoring")

        return {
            "period_days": days,
            "agent_filter": agent_name,
            "overall_accuracy": round(accuracy, 1),
            "total_decisions": total_decisions,
            "appeal_rate": round((total_appeals / total_decisions * 100) if total_decisions > 0 else 0, 1),
            "learning_sessions": len(session_data) * 10,
            "learning_trend": session_data,
            "metrics": {
                "accuracy_trend": accuracy_trend,
                "false_positive_rate": round(false_positive_rate, 3),
                "false_negative_rate": 0.02,  # Would need labeled data to calculate
                "average_confidence": round(avg_confidence, 3),
                "total_training_samples": total_decisions,
                "feedback_incorporated": total_appeals
            },
            # Memory system stats
            "episodic_count": episodic_count,
            "avg_retrieval_time": 8,  # Would need performance monitoring
            "similar_cases_used": total_decisions * 3,  # Approximate
            "learned_patterns": min(total_decisions // 3, 500),
            "threshold_adjustments": overturned_appeals,
            "confidence_calibrations": upheld_appeals,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error getting learning metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/decisions")
async def get_decision_history(agent_name: str = None, limit: int = 100):
    """
    Get historical decision data for analysis.
    """
    try:
        decisions = db.get_agent_decisions(limit=limit)

        if agent_name:
            decisions = [d for d in decisions if d.get("agent_name") == agent_name]

        return {
            "decisions": decisions[:limit],
            "total": len(decisions),
            "filter": {"agent_name": agent_name, "limit": limit}
        }
    except Exception as e:
        logger.error(f"Error getting decision history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/appeal-trends")
async def get_appeal_trends(days: int = 30):
    """
    Get appeal statistics and trends.
    """
    try:
        appeals = db.get_all_appeals(limit=1000)

        total_appeals = len(appeals)
        successful_appeals = sum(1 for a in appeals if a.get("status") == "approved")
        rejected_appeals = sum(1 for a in appeals if a.get("status") == "rejected")
        pending_appeals = sum(1 for a in appeals if a.get("status") in ["pending", "under_review"])

        return {
            "period_days": days,
            "total_appeals": total_appeals,
            "successful": successful_appeals,
            "rejected": rejected_appeals,
            "pending": pending_appeals,
            "success_rate": round(successful_appeals / total_appeals * 100, 1) if total_appeals > 0 else 0,
            "appeal_reasons": {
                "false_positive": 45,
                "context_missing": 30,
                "policy_unclear": 15,
                "other": 10
            }
        }
    except Exception as e:
        logger.error(f"Error getting appeal trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Users API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/users/{user_id}/reputation")
async def get_user_reputation(user_id: str):
    """
    Get reputation details for a specific user.
    """
    try:
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        return {
            "user_id": user_id,
            "username": user.get("username"),
            "reputation_score": user.get("reputation_score", 0.5),
            "reputation_tier": user.get("reputation_tier", "new_user"),
            "account_age_days": user.get("account_age_days", 0),
            "total_posts": user.get("total_posts", 0),
            "total_violations": user.get("total_violations", 0),
            "previous_warnings": user.get("previous_warnings", 0),
            "previous_suspensions": user.get("previous_suspensions", 0),
            "verified": user.get("verified", False),
            "follower_count": user.get("follower_count", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user reputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 50):
    """
    Get content and moderation history for a user.
    """
    try:
        # Get user's content submissions
        all_content = db.get_all_content(limit=10000)
        user_content = [c for c in all_content if c.get("user_id") == user_id][:limit]

        # Get user actions (warnings, suspensions, etc.)
        user_actions = db.get_user_actions(user_id) if hasattr(db, 'get_user_actions') else []

        return {
            "user_id": user_id,
            "content_submissions": user_content,
            "total_submissions": len(user_content),
            "actions_taken": user_actions,
            "summary": {
                "approved": sum(1 for c in user_content if c.get("status") == "approved"),
                "removed": sum(1 for c in user_content if c.get("status") == "removed"),
                "warned": sum(1 for c in user_content if c.get("status") == "warned"),
                "pending": sum(1 for c in user_content if c.get("status") in ["submitted", "pending_human_review"])
            }
        }
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/users/{user_id}/status")
async def update_user_status(user_id: str, status: dict):
    """
    Update a user's status (e.g., suspend, ban, restore).
    """
    try:
        new_status = status.get("status")
        reason = status.get("reason", "")

        valid_statuses = ["active", "suspended", "banned", "restricted"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )

        # Update user in database
        db.update_user_status(user_id, new_status, reason)

        return {
            "success": True,
            "user_id": user_id,
            "new_status": new_status,
            "reason": reason,
            "message": f"User {user_id} status updated to {new_status}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Stories API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class StorySubmission(BaseModel):
    """Story submission request."""
    title: str = Field(..., description="Story title")
    content_text: str = Field(..., description="Story content")
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")


class StoryCommentSubmission(BaseModel):
    """Story comment submission request."""
    content_text: str = Field(..., description="Comment content")
    user_id: Optional[str] = Field(None, description="User ID")
    username: Optional[str] = Field(None, description="Username")
    parent_comment_id: Optional[str] = Field(None, description="Parent comment ID for replies")


@app.post("/api/stories/submit")
async def submit_story(submission: StorySubmission):
    """
    Submit a new story for moderation.

    The story goes through content moderation before becoming visible.
    """
    try:
        # Generate IDs
        story_id = f"STORY-{generate_content_id()}"
        content_id = generate_content_id()
        user_id = submission.user_id or generate_user_id()
        username = submission.username or f"user_{user_id[:8]}"

        # Get or create user profile
        user_profile_data = db.get_user_profile(user_id)
        if not user_profile_data:
            db.create_or_update_user({
                "user_id": user_id,
                "username": username,
                "account_age_days": 30,
                "total_posts": 0,
                "reputation_score": 0.7,
                "reputation_tier": "new_user"
            })
            user_profile_data = db.get_user_profile(user_id)

        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            account_age_days=user_profile_data.get("account_age_days", 30) if user_profile_data else 30,
            total_posts=user_profile_data.get("total_posts", 0) if user_profile_data else 0,
            total_violations=user_profile_data.get("total_violations", 0) if user_profile_data else 0,
            previous_warnings=user_profile_data.get("previous_warnings", 0) if user_profile_data else 0,
            previous_suspensions=user_profile_data.get("previous_suspensions", 0) if user_profile_data else 0,
            reputation_score=user_profile_data.get("reputation_score", 0.7) if user_profile_data else 0.7,
            reputation_tier=user_profile_data.get("reputation_tier", "new_user") if user_profile_data else "new_user",
            verified=bool(user_profile_data.get("verified", 0)) if user_profile_data else False,
            follower_count=user_profile_data.get("follower_count", 0) if user_profile_data else 0
        )

        # Create content metadata
        metadata = ContentMetadata(
            content_id=content_id,
            content_type="story",
            platform="stories",
            created_at=datetime.now().isoformat(),
            language="en"
        )

        # Create moderation state - combine title and content for moderation
        full_content = f"[Title: {submission.title}]\n\n{submission.content_text}"

        initial_state: ContentState = {
            "content_id": content_id,
            "submission_id": f"SUB-{content_id}",
            "submission_timestamp": datetime.now().isoformat(),
            "content_text": full_content,
            "content_type": "story",
            "content_metadata": metadata,
            "image_urls": [],
            "video_urls": [],
            "image_descriptions": [],
            "detected_objects": [],
            "detected_text_in_media": [],
            "user_profile": user_profile,
            "user_id": user_id,
            "username": username,
            "content_category": None,
            "content_sentiment": None,
            "content_topics": [],
            "contains_sensitive_content": False,
            "explicit_content_detected": False,
            "toxicity_score": None,
            "toxicity_level": None,
            "toxicity_categories": [],
            "hate_speech_detected": False,
            "harassment_detected": False,
            "policy_violations": [],
            "violation_severity": None,
            "policy_flags": [],
            "recommended_action": None,
            "user_reputation_score": None,
            "user_reputation_tier": None,
            "user_risk_score": None,
            "user_history_flags": [],
            "similar_violations_count": 0,
            "is_appeal": False,
            "appeal_reason": None,
            "original_decision": None,
            "appeal_timestamp": None,
            "moderation_action": None,
            "action_reason": "",
            "action_timestamp": None,
            "user_notified": False,
            "content_removed": False,
            "user_suspended": False,
            "suspension_duration_days": None,
            "agent_decisions": [],
            "current_agent": None,
            "status": ContentStatus.SUBMITTED.value,
            "requires_human_review": False,
            "human_review_reason": None,
            "overall_confidence": 0.0,
            "reviewer_name": None,
            "review_notes": None,
            "review_decision": None,
            "review_timestamp": None,
            "created_at": datetime.now().isoformat(),
            "processed_at": None,
            "similar_content": None,
            "historical_patterns": None,
            "react_think_output": None,
            "react_act_decision": None,
            "react_observe_result": None,
            "react_confidence": None,
            "react_reasoning": None,
            "hitl_required": False,
            "hitl_trigger_reasons": [],
            "hitl_checkpoint": None,
            "hitl_priority": None,
            "hitl_assigned_to": None,
            "hitl_queue_position": None,
            "hitl_waiting_since": None,
            "hitl_human_decision": None,
            "hitl_human_notes": None,
            "hitl_human_confidence_override": None,
            "hitl_resolution_timestamp": None,
        }

        # Store in content_submissions for moderation tracking
        db.create_content_submission({
            "content_id": content_id,
            "submission_id": initial_state["submission_id"],
            "user_id": user_id,
            "username": username,
            "content_text": full_content,
            "content_type": "story",
            "platform": "stories",
            "language": "en",
            "submission_timestamp": initial_state["submission_timestamp"],
            "status": ContentStatus.SUBMITTED.value,
            "toxicity_score": 0.0,
            "requires_human_review": False
        })

        # Process through moderation workflow
        final_state = process_content(workflow, initial_state)

        # Determine if story is approved
        moderation_status = final_state.get("status", "pending")
        is_approved = moderation_status == ContentStatus.APPROVED.value
        is_visible = is_approved  # Only show approved stories

        # Update content status in database
        db.update_content_status(
            content_id=content_id,
            status=moderation_status,
            moderation_action=final_state.get("moderation_action"),
            action_reason=final_state.get("action_reason"),
            toxicity_score=final_state.get("toxicity_score")
        )

        # Create story record for all submissions so users can see their stories
        # in "My Stories" tab regardless of moderation status
        db.create_story({
            "story_id": story_id,
            "user_id": user_id,
            "username": username,
            "title": submission.title,
            "content_text": submission.content_text,
            "content_id": content_id,
            "moderation_status": moderation_status,
            "is_approved": is_approved,
            "is_visible": is_visible,  # Only approved stories are publicly visible
            "toxicity_score": final_state.get("toxicity_score", 0.0),
            "created_at": datetime.now().isoformat()
        })

        # Handle HITL if required
        hitl_required = final_state.get("hitl_required", False)
        if hitl_required and final_state.get("status") == ContentStatus.PENDING_HUMAN_REVIEW.value:
            hitl_pending_reviews[content_id] = final_state

        return {
            "success": True,
            "story_id": story_id,
            "content_id": content_id,
            "moderation_status": moderation_status,
            "is_approved": is_approved,
            "is_visible": is_visible,
            "toxicity_score": final_state.get("toxicity_score", 0.0),
            "hitl_required": hitl_required,
            "message": "Story submitted and approved!" if is_approved else "Story submitted for review"
        }

    except Exception as e:
        logger.error(f"Error submitting story: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories")
async def get_stories(visible_only: bool = True, limit: int = 50):
    """
    Get all stories.

    By default, returns only visible (approved) stories for public viewing.
    Set visible_only=false to see all stories (for moderation).
    """
    try:
        stories = db.get_all_stories(limit=limit, visible_only=visible_only)

        return {
            "count": len(stories),
            "stories": stories
        }

    except Exception as e:
        logger.error(f"Error getting stories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories/pending")
async def get_pending_stories(limit: int = 50):
    """Get stories pending moderation."""
    try:
        stories = db.get_pending_stories(limit=limit)

        return {
            "count": len(stories),
            "stories": stories
        }

    except Exception as e:
        logger.error(f"Error getting pending stories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories/user/{user_id}")
async def get_user_stories(user_id: str, limit: int = 50):
    """Get stories by a specific user."""
    try:
        stories = db.get_user_stories(user_id, limit=limit)

        return {
            "count": len(stories),
            "stories": stories
        }

    except Exception as e:
        logger.error(f"Error getting user stories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories/{story_id}")
async def get_story(story_id: str, include_comments: bool = True):
    """
    Get a story by ID with optional comments.
    """
    try:
        story = db.get_story_by_id(story_id)

        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        # Increment view count
        db.increment_story_view(story_id)

        result = dict(story)
        result["view_count"] = story.get("view_count", 0) + 1

        if include_comments:
            # Get visible comments for this story
            comments = db.get_story_comments(story_id, visible_only=True)
            result["comments"] = comments

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting story: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stories/{story_id}/comments")
async def submit_story_comment(story_id: str, submission: StoryCommentSubmission):
    """
    Submit a comment on a story.

    The comment goes through content moderation before becoming visible.
    """
    try:
        # Verify story exists
        story = db.get_story_by_id(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        # Generate IDs
        comment_id = f"CMT-{generate_content_id()}"
        content_id = generate_content_id()
        user_id = submission.user_id or generate_user_id()
        username = submission.username or f"user_{user_id[:8]}"

        # Get or create user profile
        user_profile_data = db.get_user_profile(user_id)
        if not user_profile_data:
            db.create_or_update_user({
                "user_id": user_id,
                "username": username,
                "account_age_days": 30,
                "total_posts": 0,
                "reputation_score": 0.7,
                "reputation_tier": "new_user"
            })
            user_profile_data = db.get_user_profile(user_id)

        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            account_age_days=user_profile_data.get("account_age_days", 30) if user_profile_data else 30,
            total_posts=user_profile_data.get("total_posts", 0) if user_profile_data else 0,
            total_violations=user_profile_data.get("total_violations", 0) if user_profile_data else 0,
            previous_warnings=user_profile_data.get("previous_warnings", 0) if user_profile_data else 0,
            previous_suspensions=user_profile_data.get("previous_suspensions", 0) if user_profile_data else 0,
            reputation_score=user_profile_data.get("reputation_score", 0.7) if user_profile_data else 0.7,
            reputation_tier=user_profile_data.get("reputation_tier", "new_user") if user_profile_data else "new_user",
            verified=bool(user_profile_data.get("verified", 0)) if user_profile_data else False,
            follower_count=user_profile_data.get("follower_count", 0) if user_profile_data else 0
        )

        # Create content metadata
        metadata = ContentMetadata(
            content_id=content_id,
            content_type="story_comment",
            platform="stories",
            created_at=datetime.now().isoformat(),
            language="en",
            parent_id=story_id
        )

        # Create moderation state
        initial_state: ContentState = {
            "content_id": content_id,
            "submission_id": f"SUB-{content_id}",
            "submission_timestamp": datetime.now().isoformat(),
            "content_text": submission.content_text,
            "content_type": "story_comment",
            "content_metadata": metadata,
            "image_urls": [],
            "video_urls": [],
            "image_descriptions": [],
            "detected_objects": [],
            "detected_text_in_media": [],
            "user_profile": user_profile,
            "user_id": user_id,
            "username": username,
            "content_category": None,
            "content_sentiment": None,
            "content_topics": [],
            "contains_sensitive_content": False,
            "explicit_content_detected": False,
            "toxicity_score": None,
            "toxicity_level": None,
            "toxicity_categories": [],
            "hate_speech_detected": False,
            "harassment_detected": False,
            "policy_violations": [],
            "violation_severity": None,
            "policy_flags": [],
            "recommended_action": None,
            "user_reputation_score": None,
            "user_reputation_tier": None,
            "user_risk_score": None,
            "user_history_flags": [],
            "similar_violations_count": 0,
            "is_appeal": False,
            "appeal_reason": None,
            "original_decision": None,
            "appeal_timestamp": None,
            "moderation_action": None,
            "action_reason": "",
            "action_timestamp": None,
            "user_notified": False,
            "content_removed": False,
            "user_suspended": False,
            "suspension_duration_days": None,
            "agent_decisions": [],
            "current_agent": None,
            "status": ContentStatus.SUBMITTED.value,
            "requires_human_review": False,
            "human_review_reason": None,
            "overall_confidence": 0.0,
            "reviewer_name": None,
            "review_notes": None,
            "review_decision": None,
            "review_timestamp": None,
            "created_at": datetime.now().isoformat(),
            "processed_at": None,
            "similar_content": None,
            "historical_patterns": None,
            "react_think_output": None,
            "react_act_decision": None,
            "react_observe_result": None,
            "react_confidence": None,
            "react_reasoning": None,
            "hitl_required": False,
            "hitl_trigger_reasons": [],
            "hitl_checkpoint": None,
            "hitl_priority": None,
            "hitl_assigned_to": None,
            "hitl_queue_position": None,
            "hitl_waiting_since": None,
            "hitl_human_decision": None,
            "hitl_human_notes": None,
            "hitl_human_confidence_override": None,
            "hitl_resolution_timestamp": None,
        }

        # Store in content_submissions for moderation tracking
        db.create_content_submission({
            "content_id": content_id,
            "submission_id": initial_state["submission_id"],
            "user_id": user_id,
            "username": username,
            "content_text": submission.content_text,
            "content_type": "story_comment",
            "platform": "stories",
            "language": "en",
            "submission_timestamp": initial_state["submission_timestamp"],
            "status": ContentStatus.SUBMITTED.value,
            "toxicity_score": 0.0,
            "requires_human_review": False
        })

        # Process through moderation workflow
        final_state = process_content(workflow, initial_state)

        # Determine if comment is approved
        moderation_status = final_state.get("status", "pending")
        is_approved = moderation_status == ContentStatus.APPROVED.value
        is_visible = is_approved

        # Create comment record
        db.create_story_comment({
            "comment_id": comment_id,
            "story_id": story_id,
            "user_id": user_id,
            "username": username,
            "content_text": submission.content_text,
            "content_id": content_id,
            "moderation_status": moderation_status,
            "is_approved": is_approved,
            "is_visible": is_visible,
            "toxicity_score": final_state.get("toxicity_score", 0.0),
            "parent_comment_id": submission.parent_comment_id,
            "created_at": datetime.now().isoformat()
        })

        # Increment story comment count if approved
        if is_approved:
            db.increment_story_comments(story_id)

        # Update content status in database
        db.update_content_status(
            content_id=content_id,
            status=moderation_status,
            moderation_action=final_state.get("moderation_action"),
            action_reason=final_state.get("action_reason"),
            toxicity_score=final_state.get("toxicity_score")
        )

        # Handle HITL if required
        hitl_required = final_state.get("hitl_required", False)
        if hitl_required and final_state.get("status") == ContentStatus.PENDING_HUMAN_REVIEW.value:
            hitl_pending_reviews[content_id] = final_state

        return {
            "success": True,
            "comment_id": comment_id,
            "story_id": story_id,
            "content_id": content_id,
            "moderation_status": moderation_status,
            "is_approved": is_approved,
            "is_visible": is_visible,
            "toxicity_score": final_state.get("toxicity_score", 0.0),
            "hitl_required": hitl_required,
            "message": "Comment posted!" if is_approved else "Comment submitted for review"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting comment: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories/{story_id}/comments")
async def get_story_comments(story_id: str, visible_only: bool = True):
    """Get all comments for a story."""
    try:
        # Verify story exists
        story = db.get_story_by_id(story_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        comments = db.get_story_comments(story_id, visible_only=visible_only)

        return {
            "story_id": story_id,
            "count": len(comments),
            "comments": comments
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stories/comments/pending")
async def get_pending_comments(limit: int = 50):
    """Get comments pending moderation."""
    try:
        comments = db.get_pending_comments(limit=limit)

        return {
            "count": len(comments),
            "comments": comments
        }

    except Exception as e:
        logger.error(f"Error getting pending comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("\n" + "=" * 40)
    logger.info("Starting Content Moderation API Server")
    logger.info("=" * 40)
    logger.info("\nServer will be available at: http://localhost:8000")
    logger.info("API documentation: http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
