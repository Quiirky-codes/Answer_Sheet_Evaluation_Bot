import google.generativeai as genai

genai.configure(api_key="ENTER_YOUR_API_KEY")


model = genai.GenerativeModel('gemini-1.5-pro-latest')


system_prompt = """
You are an answer sheet evaluator,
Given a question,
you have two inputs, one is the expected answer key or schemas to the question
and the other is the answer given by the student. You have to mark the students answer
a marks from 0 to 10 based on how correct the student's answer is compared
to the answer key. Be more lenient while marking the student's answer.
Mention all the mistakes in the student's answer and provide the correct answer.
"""

system_prompt = """
**Context:**

I am an answer sheet evaluator for an educational assessment. My goal is to accurately assess student understanding while offering constructive feedback.

**Inputs:**

1. **Question:** (Text) The actual question posed to the student.
2. **Answer Key/Schema:** (Text, List) The expected answer, including key points and potential variations. 
3. **Student Answer:** (Text) The written response provided by the student.

**Output:**

* **Score (0-10):** (Number) A numerical score reflecting the overall correctness of the student's answer relative to the answer key.  
* **Feedback:** (Text) A detailed explanation of the score, highlighting:
    * **Correct elements:** Identify any aspects of the student's answer that align with the answer key.
    * **Mistakes:** Point out specific errors or missing information in the student's response. 
    * **Clarifications:** Provide additional context or explanations for key concepts related to the question.

**Grading Approach:**

* **Focus on understanding:** Prioritize the student's demonstration of knowledge and critical thinking over strictly adhering to wording.
* **Partial credit:** Award partial points for partially correct answers or those demonstrating understanding of some key concepts.
* **Consider alternative phrasings:** Accept answers that convey the same meaning as the answer key, even if phrased differently.
"""

with open("./Anonymous/5/transcribed.txt", "r") as keyfile:
    student_answer = keyfile.read()

with open("./Answer_keys/2/transcribed.txt", "r") as keyfile:
    correct_answer = keyfile.read()

question = """
Interpret the challenges faced by traditional ANN to deal with image
and what are the building blocks of CNN.
"""


combined_query = f"""{system_prompt}\n\n\n\n 
HERE IS THE QUESTION : {str(question)} \n\n\n\n\n 
HERE IS THE ANSWER KEY: {str(correct_answer)} \n\n\n\n\n 
HERE IS THE STUDENT ANSWER: {str(student_answer)},"""


response = model.generate_content(combined_query)

response.resolve()

with open("./Anonymous/5/result.txt", "a") as f:
    f.write(response.text)
