"""Upload validation and PDF parsing. No API key, no network."""

import io

import pytest

from services.rag_service import Document, IngestError, load_pdfs, validate_uploads


def make_pdf(text: str = "Hello from a test PDF.") -> bytes:
    """A minimal one-page PDF, built without a third-party writer."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return out.getvalue()


class TestValidateUploads:
    def test_accepts_a_pdf(self):
        validate_uploads([Document("a.pdf", make_pdf())])

    def test_rejects_no_files(self):
        with pytest.raises(IngestError, match="No files"):
            validate_uploads([])

    def test_rejects_non_pdf(self):
        with pytest.raises(IngestError, match="not a PDF"):
            validate_uploads([Document("notes.txt", b"hello")])

    def test_rejects_empty_file(self):
        with pytest.raises(IngestError, match="empty"):
            validate_uploads([Document("a.pdf", b"")])

    def test_rejects_too_many_files(self, monkeypatch):
        monkeypatch.setattr("services.rag_service.MAX_FILES", 2)
        docs = [Document(f"{i}.pdf", make_pdf()) for i in range(3)]
        with pytest.raises(IngestError, match="Too many files"):
            validate_uploads(docs)

    def test_rejects_oversized_upload(self, monkeypatch):
        monkeypatch.setattr("services.rag_service.MAX_UPLOAD_BYTES", 100)
        with pytest.raises(IngestError, match="too large"):
            validate_uploads([Document("big.pdf", b"x" * 200)])

    def test_size_limit_is_across_all_files_not_per_file(self, monkeypatch):
        # Three files under the cap individually can still blow past it together.
        monkeypatch.setattr("services.rag_service.MAX_UPLOAD_BYTES", 250)
        docs = [Document(f"{i}.pdf", b"x" * 100) for i in range(3)]
        with pytest.raises(IngestError, match="too large"):
            validate_uploads(docs)


class TestLoadPdfs:
    def test_extracts_text(self):
        pages = load_pdfs([Document("greeting.pdf", make_pdf("Hello from a test PDF."))])
        assert len(pages) == 1
        assert "Hello" in pages[0].page_content

    def test_source_metadata_uses_the_uploaded_filename(self):
        # PyPDFLoader records the temp path it was handed; the user has never
        # heard of that path, so it is replaced with the real filename.
        pages = load_pdfs([Document("quarterly-report.pdf", make_pdf())])
        assert pages[0].metadata["source"] == "quarterly-report.pdf"

    def test_unreadable_pdf_raises_ingest_error(self):
        with pytest.raises(IngestError, match="Could not read"):
            load_pdfs([Document("broken.pdf", b"this is not a pdf at all")])

    def test_multiple_files_are_concatenated(self):
        pages = load_pdfs([
            Document("one.pdf", make_pdf("First document")),
            Document("two.pdf", make_pdf("Second document")),
        ])
        assert len(pages) == 2
        assert {p.metadata["source"] for p in pages} == {"one.pdf", "two.pdf"}


class TestRetrieverReceivesAPlainString:
    """Regression test for a deterministic 500 that looked like an outage.

    ``StrOutputParser`` returns a ``TextAccessor``, not a ``str``. FAISS hands
    whatever it receives straight to the embedding client, and the Google SDK
    serialises a TextAccessor into a request the API rejects with
    ``500 INTERNAL`` - every time, not intermittently. The chain therefore
    coerces to ``str`` before the retriever, and this test pins that.
    """

    def test_chain_coerces_parser_output_before_retrieval(self):
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.documents import Document as LCDocument
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from services.rag_service import build_chain

        seen: list = []

        from langchain_core.runnables import RunnableLambda

        def spy_retrieve(query):
            """Stands in for the FAISS retriever and records what it is given."""
            seen.append(query)
            return [LCDocument(page_content="context text")]

        class SpyIndex:
            def as_retriever(self, **_kwargs):
                return RunnableLambda(spy_retrieve)

        llm = FakeListChatModel(responses=["reformulated question", "the answer"])
        history = InMemoryChatMessageHistory()
        chain = build_chain(llm, SpyIndex(), 3, lambda _sid: history)

        chain.invoke({"input": "original question"},
                     config={"configurable": {"session_id": "t"}})

        assert seen, "the retriever was never called"
        assert type(seen[0]) is str, (
            f"retriever received {type(seen[0]).__name__}, not str - "
            "the coercion in build_chain has been removed"
        )
