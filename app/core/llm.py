import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_API_KEY
)


def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
Answer the question using ONLY the provided context.

DO NOT add any external knowledge.

If the answer is not in the context, say "Information not available".

Context:
{context_text}

Question:
{query}

Answer:
"""


def generate_answer(query, contexts):
    if not HF_API_KEY:
        return "Error: HF_API_KEY not set"

    prompt = build_prompt(query, contexts)

    try:
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"