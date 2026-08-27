import os
from typing import Any
from google.genai import errors
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
)

DEFAULT_MODEL = "gemini-3.6-flash"
SYSTEM_PROMPT = """
You are an enterprise policy assistant.

Answer the user's question using ONLY the supplied evidence.

Rules:

1. Use only the provided evidence for company-policy facts.

2. Every factual claim must cite the supporting evidence
   using this exact format:
   [Evidence N]

3. Never invent an evidence number.

4. If the evidence is insufficient, clearly say that
   the available evidence is insufficient.

5. If evidence conflicts, explain the conflict.

6. Answer directly and concisely.

7. Do not mention retrieval scores, RRF scores,
   reranker scores, embeddings, or internal system details.
""".strip()


class GroundedAnswerGenerator:
    """
    Generates grounded answers from the evidence packet
    created by the Context Builder.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        load_dotenv()

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing"
            )

        self.model_name = model_name

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=api_key,
            thinking_level="low",
            max_tokens=1024,
            max_retries=1,
            timeout=120,
        )

        self.prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        """
Question:
{question}

Evidence:
{context}

Answer the question using only the evidence above.
""".strip(),
                    ),
                ]
            )
        )

        # Thin LangChain orchestration:
        # Prompt → Gemini
        self.chain = (
            self.prompt
            | self.llm
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Generate one grounded answer.
        """

        question = question.strip()
        context = context.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty"
            )

        if not context:
            raise ValueError(
                "Evidence context cannot be empty"
            )

        try:
            try:
                response = self.chain.invoke(
                    {
                        "question": question,
                        "context": context,
                    }
                )

            except ChatGoogleGenerativeAIError as exc:
                message = str(exc)

                if (
                    "429" in message
                    or "RESOURCE_EXHAUSTED" in message
                ):
                    raise RuntimeError(
                        "Gemini API quota is exhausted. "
                        "Please try again when quota is available."
                    ) from exc

                raise RuntimeError(
                    "Gemini generation failed."
                ) from exc

        except errors.ServerError as exc:
            if getattr(exc, "code", None) == 504:
                raise RuntimeError(
                    "Gemini generation timed out. "
                    "Please retry the request."
                ) from exc

            raise

        answer = response.text.strip()

        if not answer:
            raise ValueError(
                "Gemini returned an empty answer"
            )

        return {
            "answer": answer,
            "model": self.model_name,
            "usage": (
                response.usage_metadata
                or {}
            ),
        }