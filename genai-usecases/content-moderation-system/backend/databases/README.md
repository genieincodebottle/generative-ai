# Databases Folder

This folder contains all database files for the Content Moderation System.

## Database Files

### 1. moderation_data.db (SQLite)
Main content moderation database containing:
- **content_submissions**: All submitted content and moderation results
- **stories**: User-submitted stories
- **story_comments**: Comments on stories
- **agent_executions**: Individual AI agent decisions
- **final_decisions**: Final moderation decisions
- **appeals**: User appeals for moderation decisions
- **manual_reviews**: Human moderator reviews
- **user_profiles**: User profiles and reputation
- **moderation_history**: Complete audit trail

### 2. moderation_auth.db (SQLite)
Authentication and user management database containing:
- **users**: User accounts with credentials and roles
- **user_sessions**: Active user sessions

### 3. chroma_moderation_db/ (ChromaDB)
Vector database for semantic memory and pattern learning:
- **moderation_patterns**: Learned moderation patterns
- **similar_cases**: Similar content examples
- **policy_examples**: Policy violation examples

## Database Initialization

Databases are automatically created when you first run:
```bash
python scripts/initialize_users.py
```

Or when the backend starts for the first time.

## Database Cleanup

To remove all data (including users):
```bash
python scripts/cleanup_data.py
```

Then reinitialize users:
```bash
python scripts/initialize_users.py
```

## Backup

To backup your databases:
```bash
# Windows
copy databases\*.db backup_folder\

# Linux/Mac
cp databases/*.db backup_folder/
```

## Migration to Other Databases

The system uses SQLite by default, but you can modify the code to use:
- PostgreSQL
- MySQL
- MongoDB
- Any other database supported by Python

Update the `database.py` and `auth_db.py` files to use your preferred database connection.
