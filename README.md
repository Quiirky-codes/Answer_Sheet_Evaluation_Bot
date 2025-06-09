# Answer Sheet Evaluation using Gemini AI

This project provides an AI-based solution for extracting and evaluating handwritten or scanned student answer sheets using Google’s Generative AI models, specifically the Gemini 1.5 Pro series. It aims to streamline academic assessments by automating text extraction, answer evaluation, and feedback generation.

# Table of Contents
  
  * Overview

  * Key Features

  * Project Structure

  * Installation

  * Configuration

  * Usage Instructions

  * Prompt Design

  * Sample Output

  * Dependencies

  * Future Improvements

# Overview

The primary objective of this project is to develop an AI-assisted application that evaluates academic answer sheets by:

  1. Extracting content from handwritten or printed images or scanned PDFs.

  2. Generating scores and feedback based on predefined answer keys using LLMs.

  3. Structuring the output in a markdown format for easy review and archival.

This system is designed for academic institutions seeking to enhance grading consistency, minimize manual effort, and facilitate the digital archiving of internal assessments.


# Key Features

* Optical character recognition and visual understanding of handwritten and printed content.

* Support for single-image input or multi-page PDF-based evaluations.

* Automatic scoring and detailed feedback generation using Google's Gemini API.

* Configurable model and prompt architecture.

* Markdown-formatted output for human verification or integration into downstream systems.

# Project Structure

```
.
├── main.py              # Script for evaluating a single answer sheet image
├── PdfToText.py         # Script to extract and evaluate answers from a scanned PDF
├── prompts.py           # Contains system prompts, model selections, and safety settings
├── student1.md          # Output file storing extracted answers and feedback
├── test.md              # Sample test paper with structured questions and diagrams
├── test2.md             # Additional annotated example
├── pyproject.toml       # Poetry dependency configuration
├── poetry.lock          # Dependency lock file
└── .env                 # Environment file to store API key securely (excluded from version control)
```

# Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

## Step 1: Clone the Repository

```
git clone https://github.com/Quiirky-codes/Answer_Sheet_Evaluation_Bot.git
cd answer-sheet-evaluation
```

## Step 2: Install Poetry (if not already installed)

```
pip install poetry
```

## Step 3: Install Project Dependencies

```
poetry install
```

# Configuration

Configuration
The system requires an API key from Google Generative AI to access Gemini models. Create a .env file in the root directory and include your key:
```
GEMINI_API_KEY=ENTER_YOUR_API_KEY_HERE
```
The application automatically loads the .env file.


# Usage Instructions

1. Evaluate a Single Image (e.g., PNG, JPEG)
Edit the image path inside main.py:
```
sample_file2 = PIL.Image.open("./Student1/1/4.png")
```
Run the script:
```
poetry run python main.py
```

This will evaluate the image and append the result to ./Student1/1/text.txt.

2. Evaluate a PDF Answer Sheet
Ensure the file student1.pdf is placed in the project root. Run:
```
poetry run python PdfToText.py
```
Each page will be processed, and the evaluated content will be appended to student1.md.


#Prompt Design

Prompts are defined in prompts.py and include:

_**image_to_text_prompt**_
Designed to extract and transcribe text from scanned or handwritten images, and describe visual diagrams when present.

_**answer_sheet_evaluation_prompt**_
Used for evaluation tasks. It follows a structured format:

**Input:**

  * Question

  * Answer Key

  * Student Answer

**Output:**

  * Score (0–10)

  * Feedback:

    * Correct points identified

    * Mistakes and missing details

    * Concept clarifications

The grading approach supports partial credit, alternate phrasings, and prioritizes conceptual understanding.

# Sample Output

("C:\Users\AMITH\OneDrive\Pictures\Screenshots\Screenshot 2025-06-08 123628.png")

The file includes:

  * Transcribed content from scanned answer sheets.

  * Detailed image descriptions (for diagrams and layouts).

  * Evaluator feedback with numerical scores and qualitative comments.

  * Explanations referencing neural network concepts, weight initialization, and CNN architectures.


# Dependencies

Dependencies
All dependencies are defined in `pyproject.toml`:
```
[tool.poetry.dependencies]
python = "^3.11"
genai = "^2.1.0"
pillow = "^10.4.0"
python-dotenv = "^1.0.1"
pdf2image = "^1.17.0"

[tool.poetry.group.dev.dependencies]
ruff = "^0.6.0"
```

# System Overview and Working

The Answer Sheet Evaluation System is designed to process and evaluate scanned student answer sheets using Google's Gemini large language models. The system takes scanned PDF or image files as input, processes them using AI prompts tailored for academic assessment, and generates structured outputs including the extracted text, feedback, scores, and visual analysis. The application is modular in design, separating the concerns of data ingestion, preprocessing, prompt handling, model communication, and output formatting.

The system operates in two primary modes: single image evaluation `(main.py)` and batch PDF evaluation `(PdfToText.py)`. In both modes, the core logic remains similar—convert visual input into AI-readable content, send it to the Gemini model with a carefully constructed prompt, and capture the structured output for review.

**1. Image and PDF Processing**
For PDFs, the system uses the pdf2image library to convert each page of the scanned document into high-resolution images. Each image typically contains handwritten text and diagrams, which are treated as input data. This is crucial because raw PDFs from scanning devices are not text-readable by default; hence, this conversion is the first preprocessing step.

For individual image files, the system directly loads the image using the Pillow library `(PIL.Image)`. It assumes the image is a clean scan or photo of the student's written answer.

**2. Prompt Construction and Model Selection**
The project includes a set of engineered prompts in `prompts.py`, which are selected based on the task. Two main prompts are used:

  * **Image-to-Text Prompt:** Extracts structured text from images, including description of diagrams, layout elements, and transcribed content.

  * **Answer Evaluation Prompt:** Compares a student's response to an ideal answer or marking scheme and provides a score along with detailed feedback.

The selected prompt, along with the image, is passed to the Gemini model using the `generate_content` method of the `GenerativeModel` class from the `google.generativeai` library. The model used by default is `gemini-1.5-pro`, which is optimized for multimodal understanding (i.e., capable of understanding both text and image inputs).

**3. AI-Powered Evaluation**
The Gemini model processes the prompt and image together, inferring context, extracting meaning from handwritten content, comparing it with the answer key, and finally producing output that includes:

  * A numeric score on a scale of 0 to 10

  * Detailed qualitative feedback

  * Highlighting correct and incorrect aspects of the answer

  * Suggestions or clarifications if needed

This output is not only accurate in terms of grading but also rich in pedagogy, supporting educators in identifying learning gaps.

**4. Output Generation**
The final output is stored in a `markdown` file (e.g., student1.md). This includes:

  * Transcribed text (question/answer pairs)

  * Extracted diagrams (described textually)

  * AI-generated scores and remarks

  * Meta-descriptions of the scanned page layout

This structured format supports easy manual review and future integration with learning management systems (LMS).

## Conceptual Block Diagram

```
                    +-------------------+
                    |  Scanned Input    |
                    | (PDF or PNG/JPG)  |
                    +-------------------+
                              |
        +---------------------+----------------------+
        |                                            |
+---------------+                           +----------------+
| PDF Conversion|                           |  Image Loader  |
| (pdf2image)   |                           | (PIL.Image)     |
+---------------+                           +----------------+
        |                                            |
        +---------------------+----------------------+
                              |
                      +---------------+
                      |  Prompt Engine |
                      | (prompts.py)   |
                      +---------------+
                              |
                      +---------------+
                      |  Gemini API    |
                      | (1.5 Pro Model)|
                      +---------------+
                              |
          +------------------------------------------+
          | Structured Output (Score, Feedback, Text)|
          +------------------------------------------+
                              |
                      +------------------+
                      | Markdown Renderer|
                      |  (student1.md)   |
                      +------------------+
```

# Example Scenario

Example Scenario
Let’s consider a typical use-case. A teacher scans answer booklets from a class and saves them as a PDF file. The file is processed by `PdfToText.py`, which converts each page into an image and passes it to the Gemini model. The model evaluates each page using prompts that simulate how a human teacher would grade, including assigning partial marks, offering feedback, and commenting on diagrams. The results are stored in a markdown file that can be reviewed, shared, or uploaded to an academic database.

This pipeline ensures that evaluation is not only accurate but also consistent across large batches of papers, eliminating human fatigue or subjectivity.




