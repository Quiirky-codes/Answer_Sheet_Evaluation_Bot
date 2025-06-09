import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# helper files
from prompts import (
    image_to_text_prompt,
    answer_sheet_evaluation_prompt,
    safety_settings,
    models_available,
)

# Configure the API Key
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name=models_available[7], safety_settings=safety_settings
)

app = Flask(__name__)
CORS(app)


@app.route("/api/process", methods=["POST"])
def process():
    print("Running Inference")
    data = request.json
    if data is None:
        return jsonify({"error": "No data provided"}), 400
    question = data.get("question")
    answer_key = data.get("answerKey")
    student_answer = data.get("studentAnswer")

    combined_query = f"""{answer_sheet_evaluation_prompt} 
    HERE IS THE QUESTION : {str(question)} 
    HERE IS THE ANSWER KEY: {str(answer_key)}
    HERE IS THE STUDENT ANSWER: {str(student_answer)}
    """

    response = model.generate_content(combined_query)
    response.resolve()
    response_text = response.text

    return jsonify({"markdown": response_text})


@app.route("/api/pdf-to-text", methods=["POST"])
def pdf_to_text():
    data = request.json
    if data is None:
        return jsonify({"error": "No data provided"}), 400
    pdf_file = data.get("pdfFile")
    response = model.generate_content(pdf_file)
    response.resolve()
    response_text = response.text

    return jsonify({"markdown": response_text})


if __name__ == "__main__":
    app.run(port=5000)
