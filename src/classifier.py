"""Query classification logic."""

from config import QueryCategory
from prompts import get_classifier_prompt


class QueryClassifier:
    """Classifies user queries through the configured LLM service."""

    def __init__(self, llm_service) -> None:
        self.llm_service = llm_service

    def classify(self, query: str, chat_history: str = "") -> str:
        """Return one supported query category."""

        result = self.llm_service.call_llm(
            [{"role": "user", "content": get_classifier_prompt(query, chat_history)}],
            temp=0.0,
            max_tokens=10,
        )
        result = result.strip().upper()
        valid = {
            QueryCategory.ACADEMIC,
            QueryCategory.GREETING,
            QueryCategory.META,
            QueryCategory.GENERAL,
            QueryCategory.INJECTION,
        }
        return result if result in valid else QueryCategory.GENERAL
