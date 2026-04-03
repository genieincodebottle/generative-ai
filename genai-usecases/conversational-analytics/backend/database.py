import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not set in environment variables")

client = MongoClient(MONGODB_URI)

# Feedback database & collection
feedback_db = client.sentiment
feedback_collection = feedback_db.sentiment_analytics
