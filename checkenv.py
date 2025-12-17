import os
from dotenv import load_dotenv
load_dotenv()
print("API Key:", os.getenv("ROBOFLOW_API_KEY"))