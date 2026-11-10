import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from pydantic import BaseModel, Field


DEFAULT_MODEL = "gemini-3.6-flash"


class AnswerJudgeSchema(BaseModel):
    fact_coverage_flags: list[bool] = Field(
        description=(
            "One boolean for each expected fact, "
            "in exactly the same order. True means "
            "the answer clearly conveys that fact."
        )
    )

    contradiction_found: bool = Field(
        description=(
            "True if the answer contradicts an "
            "expected fact or improperly strengthens "
            "a conditional rule into an unconditional one."
        )
    )

    contradiction_explanation: list[str] = Field(
        default_factory=list,
        description=(
            "Brief explanations of contradictions "
            "or material overstatements."
        ),
    )

    unsupported_or_overstated_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims that materially overstate or "
            "conflict with the reference facts. "
            "Do not list harmless additional detail."
        ),
    )

    answerability_handled_correctly: bool = Field(
        description=(
            "For answerable questions, the answer "
            "should actually answer the question. "
            "For unanswerable questions, it should "
            "avoid inventing the requested exact fact."
        )
    )

    rationale: str = Field(
        description=(
            "A short explanation of the judgment."
        )
    )


SYSTEM_PROMPT = """
You are evaluating the factual quality of a RAG answer.

You will receive:

1. A user question.
2. Whether the benchmark says the question is answerable.
3. A list of expected facts.
4. The generated answer.

Evaluate only against the supplied benchmark information.

Rules:

- A fact is covered if the answer communicates the same
  meaning, even with different wording.
- Do not require exact wording.
- Do not penalize harmless extra explanation.
- Mark contradiction_found=true when the answer conflicts
  with an expected fact.
- Also treat an unjustified strengthening as a contradiction.
  Example:
  expected: "X happens when condition Y applies"
  answer: "X always happens"
  This is an overstatement.
- Do not assume that an additional claim is false merely
  because it is absent from expected_facts.
- Put a claim in unsupported_or_overstated_claims only when
  it clearly conflicts with, changes, or materially
  overstates the supplied reference facts.
- Ignore citation formatting when deciding fact coverage.
- If answerable=false, the answer must not invent the exact
  information that the benchmark says cannot be determined.
- fact_coverage_flags must contain exactly one boolean for
  every expected fact and must preserve their order.

Be strict but semantic.
""".strip()


class AnswerQualityJudge:
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
            temperature=0,
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

Benchmark answerable:
{answerable}

Expected facts:
{expected_facts}

Generated answer:
{answer}

Evaluate this answer.
""".strip(),
                    ),
                ]
            )
        )

        structured_llm = (
            self.llm.with_structured_output(
                AnswerJudgeSchema
            )
        )

        self.chain = (
            self.prompt
            | structured_llm
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        expected_facts: list[str],
        answerable: bool,
    ) -> dict[str, Any]:
        question = question.strip()
        answer = answer.strip()

        if not question:
            raise ValueError(
                "Judge question cannot be empty"
            )

        if not answer:
            raise ValueError(
                "Judge answer cannot be empty"
            )

        if not isinstance(
            answerable,
            bool,
        ):
            raise ValueError(
                "answerable must be a boolean"
            )

        cleaned_facts = [
            fact.strip()
            for fact in expected_facts
            if isinstance(fact, str)
            and fact.strip()
        ]

        formatted_facts = "\n".join(
            f"{index}. {fact}"
            for index, fact in enumerate(
                cleaned_facts,
                start=1,
            )
        )

        if not formatted_facts:
            formatted_facts = (
                "(No explicit expected facts)"
            )

        result = self.chain.invoke(
            {
                "question": question,
                "answerable": answerable,
                "expected_facts": (
                    formatted_facts
                ),
                "answer": answer,
            }
        )

        if isinstance(
            result,
            AnswerJudgeSchema,
        ):
            judged = result.model_dump()
        else:
            judged = dict(result)

        flags = judged[
            "fact_coverage_flags"
        ]

        if len(flags) != len(
            cleaned_facts
        ):
            raise ValueError(
                "Judge returned the wrong number "
                "of fact coverage flags. "
                f"Expected {len(cleaned_facts)}, "
                f"received {len(flags)}."
            )

        if cleaned_facts:
            fact_coverage_score = (
                sum(flags)
                / len(cleaned_facts)
            )
        else:
            fact_coverage_score = None

        return {
            "model": self.model_name,
            "fact_count": len(
                cleaned_facts
            ),
            "facts_covered": sum(
                flags
            ),
            "fact_coverage_score": (
                fact_coverage_score
            ),
            "fact_coverage_flags": flags,
            "contradiction_found": judged[
                "contradiction_found"
            ],
            "contradiction_explanation": judged[
                "contradiction_explanation"
            ],
            "unsupported_or_overstated_claims": (
                judged[
                    "unsupported_or_overstated_claims"
                ]
            ),
            "answerability_handled_correctly": (
                judged[
                    "answerability_handled_correctly"
                ]
            ),
            "rationale": judged[
                "rationale"
            ],
        }