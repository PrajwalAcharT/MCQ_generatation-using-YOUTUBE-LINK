from chromadb.config import Settings

# Configure settings for the Chroma database
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duck+parquet",  # Specifies database type
    persist_directory="db",          # Directory where the Chroma database will be stored
    anonymized_telemetry=False       # Disable telemetry for privacy
)