import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def get_llm():
    """
    Initialize and return the LLM based on LLM_PROVIDER env var.
    Supported: 'groq' (primary), 'gemini' (secondary).
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set in environment variables. "
                "Set it or switch LLM_PROVIDER to 'gemini'."
            )
        from langchain_groq import ChatGroq
        logger.info("Using Groq LLM provider (llama-3.3-70b-versatile)")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=api_key,
        )

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set in environment variables. "
                "Set it or switch LLM_PROVIDER to 'groq'."
            )
        os.environ["GOOGLE_API_KEY"] = api_key
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Using Gemini LLM provider (gemini-2.0-flash)")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
        )

    raise ValueError(
        f"Invalid LLM_PROVIDER: '{provider}'. Must be 'groq' or 'gemini'."
    )


ANALYZE_PROMPT = PromptTemplate(
    input_variables=["feedback"],
    template=(
        "Analyze the following customer feedback:\n"
        "{feedback}\n\n"
        "Provide the following as per given format:\n"
        "1. Key topics (comma-separated)\n"
        "2. Sentiment (If mixed sentiment then must be separated by pipe in bracket)\n"
        "3. Emerging trends (comma-separated)"
    ),
)


def build_analyze_chain():
    """Build the LCEL analysis chain: prompt -> llm -> string output."""
    llm = get_llm()
    return ANALYZE_PROMPT | llm | StrOutputParser()
