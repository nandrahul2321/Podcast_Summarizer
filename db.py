# db.py
from pymongo import MongoClient, ASCENDING

client = MongoClient('mongodb://localhost:27017/')
db = client['Podcast_Summarizer_5']
users_collection = db['users']
summaries_collection = db['summaries']
history_collection = db['history']
comments_collection = db['comments']
chat_history_collection = db['chat_history']

# -------------------- Indexes --------------------
# Drop stale multilanguage index if it exists from prior runs
try:
    summaries_collection.drop_index('idx_video_lang')
except Exception:
    pass

# Simple index on video_id for fast lookups
summaries_collection.create_index(
    [('video_id', ASCENDING)],
    name='idx_video_id'
)

# Index for fast comment retrieval by video
comments_collection.create_index(
    [('video_id', ASCENDING), ('created_at', ASCENDING)],
    name='idx_comments_video'
)

# Index for chat history retrieval
chat_history_collection.create_index(
    [('user_id', ASCENDING), ('video_id', ASCENDING), ('timestamp', ASCENDING)],
    name='idx_chat_history'
)