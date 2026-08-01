import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Try all available models to find one with remaining quota
models = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

print("Testing which model has remaining free-tier quota...\n")
for m in models:
    try:
        r = c.models.generate_content(model=m, contents="Say hello in 3 words")
        print(f"  [OK]   {m}: {r.text.strip()[:50]}")
    except Exception as e:
        code = "429" if "429" in str(e) else "ERR"
        print(f"  [FAIL] {m}: {code} - quota exhausted or unavailable")
