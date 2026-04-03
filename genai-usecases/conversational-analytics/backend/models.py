from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from auth import UserRole


class User(BaseModel):
    username: str
    password: str
    role: UserRole = UserRole.USER

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


class UserOut(BaseModel):
    username: str
    role: UserRole


class Token(BaseModel):
    access_token: str
    token_type: str


class FeedbackSubmission(BaseModel):
    feedback: str
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackResponse(BaseModel):
    current: List[FeedbackSubmission]
    historical: List[FeedbackSubmission]
