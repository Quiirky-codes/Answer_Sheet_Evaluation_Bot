import PIL.Image

from IPython.display import Markdown

import google.generativeai as genai
from prompts import system_prompt

# access api key from .env file
from dotenv import load_dotenv
import os

load_dotenv()
env = os.environ


genai.configure(api_key="ENTER_YOUR_API_KEY")


# genai.configure(env.get("GENAI_API_KEY"))


# correct_answer = PIL.Image.open('./Aishwarya/1/MarkingSchema.png')
# student_answer_img2 = PIL.Image.open('./Aishwarya/1/2.png')
# student_answer_img3 = PIL.Image.open('./Aishwarya/1/3.png')
# student_answer_img4 = PIL.Image.open('./Aishwarya/1/4.png')


# models/gemini-1.0-pro-latest
# models/gemini-1.0-pro
# models/gemini-pro
# models/gemini-1.0-pro-001
# models/gemini-1.0-pro-vision-latest
# models/gemini-pro-vision
# models/gemini-1.5-pro-latest
# models/gemini-1.5-pro-001
# models/gemini-1.5-pro
# models/gemini-1.5-pro-exp-0801
# models/gemini-1.5-flash-latest
# models/gemini-1.5-flash-001
# models/gemini-1.5-flash
# models/gemini-1.5-flash-001-tuning
# model = genai.GenerativeModel("gemini-1.0-pro-vision-latest")

sample_file2 = PIL.Image.open("./Student1/1/4.png")


# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Prompt the model with text and the previously uploaded image.
response = model.generate_content([sample_file2, f"{system_prompt}"])

print(str(Markdown(">" + response.text).data))

with open("./Student1/1/text.txt", "a") as f:
    f.write(response.candidates[0].content.parts[0].text)
