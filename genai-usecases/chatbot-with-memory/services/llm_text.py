"""Read text out of an LLM response, whatever shape it arrives in.

Current Gemini 3 models return ``message.content`` as a **list of content
blocks**::

    [{"type": "text", "text": "the answer", "extras": {"signature": "..."}}]

Older models (Gemini 2.5 and earlier) return a plain ``str``. Code written
against the string contract breaks in two ways when the model is upgraded, and
neither announces itself:

* ``response.content.strip()`` / ``.lower()`` / ``.find()`` raise
  ``AttributeError`` - a hard crash.
* ``len(response.content)`` silently measures the *number of blocks*, so a
  900-character answer has "length 1" and fails every length check.
* Rendering it shows the raw block repr, base64 thinking signature and all.

Always route response text through ``message_text``.
"""

from __future__ import annotations


def message_text(response) -> str:
    """Return the plain text of an LLM response.

    Accepts a message object, a raw ``content`` value, a list of content
    blocks, or a string, and always returns a string.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)

    if content is None:
        return ""

    return str(content)
