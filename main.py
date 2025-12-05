import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file with OPENAI_API_KEY=your_key_here"
    )

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are a friendly Python tutor for beginners.

Your goals:
- Explain Python concepts clearly in simple language.
- Give short, correct code examples with comments.
- Create small practice exercises.
- Give encouraging feedback.

Always respond using this structure:

Concept Explanation:
[Explain the concept or answer the question in simple terms.]

Code Example:
[Provide a short, commented Python example related to the question.]

Practice Exercise:
[Give the user 1 small exercise they can try.]

Feedback:
[If the user gave code, give feedback on it.
If they did not give code, encourage them to try the exercise.]
"""


def detect_mode(user_input: str) -> str:
    """
    Very simple rule-based mode detection.
    This is just to show that you thought about user interaction.
    """
    text = user_input.lower()

    if "explain" in text or "what is" in text:
        return "explain"
    elif "example" in text or "show me" in text:
        return "example"
    elif "exercise" in text or "practice" in text or "problem" in text:
        return "exercise"
    elif "error" in text or "traceback" in text or "doesn't work" in text or "doesnt work" in text:
        return "debug"
    else:
        return "feedback"


def build_messages(user_input: str) -> list:
    """
    Build the messages list we send to the LLM.
    We include the mode so the model knows what type of help to give.
    """
    mode = detect_mode(user_input)

    mode_instruction = ""
    if mode == "explain":
        mode_instruction = "Focus on explaining the concept simply, like teaching a beginner."
    elif mode == "example":
        mode_instruction = "Focus on giving a clear, short code example with comments."
    elif mode == "exercise":
        mode_instruction = "Focus on giving 1–2 small practice tasks, do not fully solve them."
    elif mode == "debug":
        mode_instruction = "Look for mistakes in the code, explain what is wrong, then show a fixed version."
    elif mode == "feedback":
        mode_instruction = "Give helpful, encouraging feedback and suggest what the student should try next."

    system_content = SYSTEM_PROMPT + "\n\nExtra instructions for this turn: " + mode_instruction

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input},
    ]
    return messages


def ask_tutor(user_input: str) -> str:
    """
    Send the user's input to the LLM and return the tutor's response.
    Also prints token usage so you can discuss cost/tokenization in your report.
    """
    messages = build_messages(user_input)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # you can change this to gpt-3.5-turbo if required
        messages=messages,
    )

    # Get text response
    answer = response.choices[0].message.content

    # Token usage for your report
    usage = response.usage
    print(
        f"\n(Token usage: prompt={usage.prompt_tokens}, "
        f"completion={usage.completion_tokens}, total={usage.total_tokens})"
    )

    return answer


def chat_loop():
    """
    Simple console chat loop.
    """
    print("Welcome to your Python AI Tutor! Type 'quit' or 'exit' to stop.")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ("quit", "exit"):
            print("Goodbye! Keep practicing Python :)")
            break

        answer = ask_tutor(user_input)
        print("\nTutor:\n" + answer)


if __name__ == "__main__":
    chat_loop()
