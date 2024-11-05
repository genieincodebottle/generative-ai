from anthropic import Anthropic
import base64
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in environment variables")
os.environ["API_KEY"] = ANTHROPIC_API_KEY

# While PDF support is in beta, you must pass in the correct beta header
client = Anthropic(default_headers={
    "anthropic-beta": "pdfs-2024-09-25"
  }
)
# For now, only claude-3-5-sonnet-20241022 supports PDFs
MODEL_NAME = "claude-3-5-sonnet-20241022"


# Start by reading in the PDF and encoding it as base64
file_name = "pdf_3.pdf"
with open(file_name, "rb") as pdf_file:
  binary_data = pdf_file.read()
  base64_encoded_data = base64.standard_b64encode(binary_data)
  base64_string = base64_encoded_data.decode("utf-8")


prompt = """
Please do the following:
1. Extract all the content from the pdf
"""
messages = [
    {
        "role": 'user',
        "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64_string}},
            {"type": "text", "text": prompt}
        ]
    }
]

def get_completion(client, messages):
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        messages=messages
    ).content[0].text


completion = get_completion(client, messages)
print(completion)