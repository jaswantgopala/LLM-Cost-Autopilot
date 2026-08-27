import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

print(f"GROQ_API_KEY loaded: {bool(groq_key)}, starts with: {groq_key[:6] if groq_key else 'None'}")
print(f"GEMINI_API_KEY loaded: {bool(gemini_key)}, starts with: {gemini_key[:6] if gemini_key else 'None'}")