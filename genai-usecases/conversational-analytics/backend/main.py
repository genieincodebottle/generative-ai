import os
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Load environment variables BEFORE importing local modules
# (auth.py, database.py read env vars at module level)
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    UserRole,
    admin_required,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
    users_collection,
)
from database import feedback_collection
from llm_config import build_analyze_chain
from models import (
    FeedbackResponse,
    FeedbackSubmission,
    Token,
    User,
    UserOut,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Conversational Analytics API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT settings
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set in environment variables")

# Build LLM analysis chain
analyze_chain = build_analyze_chain()


# ──────────────────────────── Auth Routes ────────────────────────────


@app.post("/register", response_model=dict)
async def register(user: User, admin_key: str = Header(None, alias="X-Admin-Key")):
    """Register a new user. Pass X-Admin-Key header to create an admin."""
    logger.info("Register request for user: %s", user.username)
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    user_dict = user.model_dump()
    user_dict["password"] = get_password_hash(user.password)
    user_dict["role"] = UserRole.ADMIN if (admin_key and admin_key == SECRET_KEY) else UserRole.USER

    result = users_collection.insert_one(user_dict)
    return {"message": "User registered successfully", "id": str(result.inserted_id)}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return a JWT access token."""
    logger.info("Login attempt for user: %s", form_data.username)
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": str(access_token), "token_type": "bearer"}


@app.get("/users/me", response_model=UserOut)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserOut(username=current_user["username"], role=current_user["role"])


@app.get("/admin")
async def admin_only(current_user: dict = Depends(admin_required)):
    """Admin-only endpoint."""
    return {"message": "Welcome, admin!", "username": current_user["username"]}


# ──────────────────────────── Feedback Routes ────────────────────────────


@app.post("/feedback")
async def submit_feedback(
    feedback_item: FeedbackSubmission,
    current_user: dict = Depends(get_current_user),
):
    """Submit customer feedback."""
    logger.info("Feedback submitted by user: %s", current_user["username"])
    feedback_doc = feedback_item.model_dump()
    feedback_doc["username"] = current_user["username"]
    feedback_collection.insert_one(feedback_doc)
    return {"message": "Feedback submitted successfully"}


@app.get("/feedback", response_model=FeedbackResponse)
async def get_feedback(current_user: dict = Depends(get_current_user)):
    """Get recent (last 1 hour) and historical feedback."""
    one_hour_ago = datetime.now() - timedelta(hours=1)
    projection = {"_id": 0, "feedback": 1, "timestamp": 1}

    current_feedback = list(
        feedback_collection.find({"timestamp": {"$gte": one_hour_ago}}, projection)
    )
    historical_feedback = list(
        feedback_collection.find({"timestamp": {"$lt": one_hour_ago}}, projection)
        .sort("timestamp", -1)
        .limit(10)
    )
    return {
        "current": [FeedbackSubmission(**item) for item in current_feedback],
        "historical": [FeedbackSubmission(**item) for item in historical_feedback],
    }


# ──────────────────────────── Analytics Route ────────────────────────────


@app.get("/analytics")
async def get_analytics(current_user: dict = Depends(admin_required)):
    """Analyze all feedback using LLM. Admin-only."""
    logger.info("Analytics requested by admin: %s", current_user["username"])
    all_feedback = list(feedback_collection.find({}, {"_id": 0, "feedback": 1}))
    combined_feedback = " ".join([f["feedback"] for f in all_feedback])

    if not combined_feedback:
        raise HTTPException(status_code=404, detail="No feedback data available")

    result = analyze_chain.invoke({"feedback": combined_feedback})

    lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
    topics = lines[0] if len(lines) > 0 else ""
    sentiment = lines[1] if len(lines) > 1 else ""
    trends = lines[2] if len(lines) > 2 else ""

    return {
        "topics": [t.strip() for t in topics.split(",") if t.strip()],
        "sentiment": sentiment,
        "trends": [t.strip() for t in trends.split(",") if t.strip()],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
