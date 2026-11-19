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
   Related terminology alone does not establish the requested fact.
   Do not infer exact amounts, deadlines, or entity-specific requirements
   from general policy. Answer supported parts and identify what is missing.

5. If evidence conflicts, explain the conflict.

6. Answer directly and concisely.
   Include relevant responsibilities, conditions, and exceptions stated in
   the evidence; check that each part of the question has been addressed.

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
        model_name: str | None = None,
    ) -> None:
        load_dotenv()

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing"
            )

        # Explicit model selection avoids hardcoding a single busy model.
        # Changing model creates a separate answer-evaluation experiment.
        self.model_name = (model_name or os.getenv("GEMINI_GENERATION_MODEL", DEFAULT_MODEL)).strip()
        if not self.model_name:
            raise ValueError("Generation model name cannot be empty")

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

                if "503" in message or "UNAVAILABLE" in message:
                    raise RuntimeError(
                        "Gemini generation model is temporarily unavailable "
                        "(HTTP 503, high demand). No answer was generated. "
                        "Retry later or explicitly select an available model."
                    ) from exc

                if "504" in message or "DEADLINE_EXCEEDED" in message:
                    raise RuntimeError(
                        "Gemini generation request timed out (HTTP 504). "
                        "No answer was captured."
                    ) from exc

                raise RuntimeError(
                    "Gemini generation failed. No answer was captured."
                ) from exc

        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise RuntimeError(
                    "Gemini generation API quota is exhausted. No answer was captured."
                ) from exc
            raise
        except errors.ServerError as exc:
            code = getattr(exc, "code", None)
            if code == 503:
                raise RuntimeError(
                    "Gemini generation model is temporarily unavailable "
                    "(HTTP 503, high demand). No answer was generated. "
                    "Retry later or explicitly select an available model."
                ) from exc
            if code == 504:
                raise RuntimeError(
                    "Gemini generation request timed out (HTTP 504). "
                    "No answer was captured."
                ) from exc
            if code in (500, 502):
                raise RuntimeError(
                    f"Gemini generation service error (HTTP {code}). "
                    "No answer was captured."
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