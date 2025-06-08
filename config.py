from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "o4-mini-2025-04-16")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SQL_ROW_LIMIT = 5_000
EXECUTION_TIMEOUT_S = 300
MAX_RETURN_ROWS_CHAT = 200
MAX_ADK_RETRIES = 2
