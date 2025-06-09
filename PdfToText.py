import os
from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_path
import time

# helper files
from prompts import system_prompt, safety_settings, models_available

# Configure the API Key
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name=models_available[7], safety_settings=safety_settings
)

images = convert_from_path("./student1.pdf")

for page in images:
    tries = 1
    while tries > 0:
        try:
            response = model.generate_content(
                [page, f"{system_prompt}"], safety_settings=safety_settings
            )
            with open("student1.md", "a") as f:
                f.write(response.candidates[0].content.parts[0].text)
            tries = 0
        except Exception as e:
            print(e)
            tries -= 1
            time.sleep(30)
    time.sleep(30)  # Sleep for 5 seconds to avoid rate limiting
