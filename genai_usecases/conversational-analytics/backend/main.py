import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
from pydantic import BaseModel, Field

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

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not set in environment variables")

client = MongoClient(MONGODB_URI)
db = client.sentiment
feedback_collection = db.sentiment_analytics

# JWT settings
SECRET_KEY = os.environ.get("SECRET_KEY")  # Set this securely in production
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set in environment variables")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment variables")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Pydantic models
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema: Any) -> None:
        field_schema.update(type="string")

class User(BaseModel):
    username: str
    password: str
    role: UserRole = UserRole.USER

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserOut(BaseModel):
    username: str
    role: UserRole

    class Config:
        populate_by_name = True

class Token(BaseModel):
    access_token: str
    token_type: str

class FeedbackSubmission(BaseModel):
    feedback: str
    timestamp: datetime = Field(default_factory=datetime.now)

class AnalyticsResponse(BaseModel):
    topics: List[str]
    sentiment: Dict[str, float]
    trends: List[str]

class FeedbackResponse(BaseModel):
    current: List[FeedbackSubmission]
    historical: List[FeedbackSubmission]

class AnalysisResult(BaseModel):
    topics: List[str]
    trends: List[str]

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, convert_system_message_to_human=True)
output_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

analyze_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="Analyze the following customer feedback: \
            \n{feedback}\n\nProvide the following as per given format: \
            \n1. Key topics (comma-separated) \
            \n2. Sentiment ( If mixed sentiment then must be separated by pipe in bracket) \
            \n3. Emerging trends (comma-separated)"
)

analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt)

# API Routes
@app.post("/register", response_model=dict)
async def register(user: User, admin_key: str = Header(None, alias="X-Admin-Key")):
    """
    Register a new user.
    """
    logger.info(f"Entered in 'register' function")
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed_password

    if admin_key and admin_key == SECRET_KEY:
        user_dict["role"] = UserRole.ADMIN
    else:
        user_dict["role"] = UserRole.USER
    
    result = users_collection.insert_one(user_dict)
    return {"message": "User registered successfully", "id": str(result.inserted_id)}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and provide access token.
    """
    logger.info(f"Entered in 'login_for_access_token' function")
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    return {"access_token": str(access_token), "token_type": "bearer"}

@app.get("/users/me", response_model=UserOut)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Get current user information.
    """
    logger.info(f"Entered in 'read_users_me' function")
    return UserOut(username=current_user["username"], role=current_user["role"])

@app.get("/admin")
async def admin_only(current_user: dict = Depends(admin_required)):
    """
    Admin-only endpoint.
    """
    logger.info(f"Entered in 'admin_only' function")
    return {"message": "Welcome, admin!", "username": current_user["username"]}

@app.post("/feedback")
async def submit_feedback(feedback_item: FeedbackSubmission, current_user: dict = Depends(get_current_user)):
    """
    Submit feedback.
    """
    logger.info(f"Entered in 'submit_feedback' function")
    feedback_doc = feedback_item.dict()
    feedback_doc["username"] = current_user["username"]
    feedback_collection.insert_one(feedback_doc)
    return {"message": "Feedback submitted successfully"}

@app.get("/feedback", response_model=FeedbackResponse)
async def get_feedback(current_user: dict = Depends(get_current_user)):
    """
    Get recent and historical feedback.
    """
    logger.info(f"Entered in 'get_feedback' function")
    current_time = datetime.now()
    one_hour_ago = current_time - timedelta(hours=1)
    
    current_feedback = list(feedback_collection.find(
        {"timestamp": {"$gte": one_hour_ago}},
        {"_id": 0, "feedback": 1, "timestamp": 1}
    ))
    historical_feedback = list(feedback_collection.find(
        {"timestamp": {"$lt": one_hour_ago}},
        {"_id": 0, "feedback": 1, "timestamp": 1}
    ).sort("timestamp", -1).limit(10))
    
    return {
        "current": [FeedbackSubmission(**item) for item in current_feedback],
        "historical": [FeedbackSubmission(**item) for item in historical_feedback]
    }

@app.get("/analytics")
async def get_analytics(current_user: dict = Depends(admin_required)):
    """
    Get analytics from feedback data.
    """
    logger.info(f"Entered in 'get_analytics' function")
    all_feedback = list(feedback_collection.find({}, {"_id": 0, "feedback": 1}))
    combined_feedback = " ".join([f["feedback"] for f in all_feedback])

    if not combined_feedback:
        raise HTTPException(status_code=404, detail="No feedback data available")

    result = analyze_chain.run(feedback=combined_feedback)
    
    topics, sentiment, trends = result.split("\n")
    
    return {
        "topics": [topic.strip() for topic in topics.split(",")],
        "sentiment": sentiment.strip(),
        "trends": [trend.strip() for trend in trends.split(",")]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)