"""Chat and vision service logic. No API key, no network."""

import io

import pytest

from services.llama_service import (
    InvalidInput,
    MissingKey,
    Reply,
    chat,
    describe_image,
    encode_image,
)


def png_bytes(size=(8, 8)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


class TestEncodeImage:
    def test_accepts_a_real_png(self):
        encoded, media = encode_image(png_bytes(), "image/png")
        assert encoded and media == "image/png"

    def test_tolerates_a_charset_suffix(self):
        # Clients sometimes send "image/png; charset=binary".
        _, media = encode_image(png_bytes(), "image/png; charset=binary")
        assert media == "image/png"

    @pytest.mark.parametrize(
        "content_type", ["application/pdf", "text/plain", "", "video/mp4"]
    )
    def test_rejects_unsupported_types(self, content_type):
        with pytest.raises(InvalidInput, match="not a supported image type"):
            encode_image(png_bytes(), content_type)

    def test_rejects_empty(self):
        with pytest.raises(InvalidInput, match="empty"):
            encode_image(b"", "image/png")

    def test_rejects_oversized(self, monkeypatch):
        monkeypatch.setattr("services.llama_service.MAX_IMAGE_BYTES", 10)
        with pytest.raises(InvalidInput, match="too large"):
            encode_image(png_bytes(), "image/png")

    def test_rejects_a_file_that_is_not_really_an_image(self):
        """A .png name and an image/png header prove nothing.

        Catching it here gives a clear message instead of a provider-side
        error that explains nothing.
        """
        with pytest.raises(InvalidInput, match="not a readable image"):
            encode_image(b"this is definitely not a png", "image/png")


class TestChatGuards:
    def test_rejects_no_messages(self):
        with pytest.raises(InvalidInput, match="No messages"):
            chat([], "meta-llama/llama-4-scout-17b-16e-instruct")

    def test_rejects_unknown_model(self):
        with pytest.raises(InvalidInput, match="Unknown model"):
            chat([{"role": "user", "content": "hi"}], "not-a-model")

    def test_missing_key_raises_rather_than_returning_a_string(self, monkeypatch):
        """The original returned "Groq API key not set." as the answer.

        The caller could not tell that apart from something the model said,
        so it was rendered in the chat window as a reply.
        """
        monkeypatch.setattr("services.llama_service.api_key", lambda n: None)
        with pytest.raises(MissingKey, match="GROQ_API_KEY"):
            chat([{"role": "user", "content": "hi"}],
                 "meta-llama/llama-4-scout-17b-16e-instruct")


class TestVisionGuards:
    def test_rejects_a_non_vision_model(self):
        with pytest.raises(InvalidInput, match="does not accept images"):
            describe_image(png_bytes(), "image/png", "read this",
                           "llama-3.3-70b-versatile")

    def test_rejects_an_empty_prompt(self):
        with pytest.raises(InvalidInput, match="prompt is empty"):
            describe_image(png_bytes(), "image/png", "   ",
                           "meta-llama/llama-4-scout-17b-16e-instruct")


class TestReply:
    def test_fallback_is_reported_not_hidden(self):
        # The caller must always be able to tell which provider answered.
        reply = Reply(text="hi", model="gemini-flash-latest", provider="google",
                      fallback_used=True, fallback_reason="primary timed out")
        assert reply.fallback_used is True
        assert reply.provider == "google"
        assert reply.fallback_reason
