"""Text cleaning and source formatting helpers."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize retrieved text while preserving the original app behavior."""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"(Page\s*\d+|Date\s*\d+/\d+/\d+)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\|\~\^\`]{2,}", " ", text)

    replacements = {
        r"\bl\b": "1",
        r"\bO\b": "0",
        r"--+": "—",
        r"\s+[-–—]\s+": " — ",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def format_source(metadata: dict) -> str:
    """Format document metadata into the source label shown in the UI."""

    parts = [metadata.get(key, "") for key in ["subject", "module"] if metadata.get(key)]
    if semester := metadata.get("semester"):
        parts.append(f"Semester {semester}")
    return " • ".join(parts) if parts else "Unknown Source"
