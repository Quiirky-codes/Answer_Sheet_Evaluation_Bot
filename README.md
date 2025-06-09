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

<!-- poppler is required -->
