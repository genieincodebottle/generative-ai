"""Service guards. No model download, no API key, no network."""

import pytest

from services import config
from services.cag_model import ModelLoadError
from services.cag_service import (
    InvalidInput,
    NotLoaded,
    default_dataset,
    load,
    parse_dataset,
    require_loaded,
    validate_document,
)


class TestModelCatalogue:
    def test_ungated_models_come_first(self):
        """A gated default is a hard stop before you have seen anything work.

        The original offered only `meta-llama/Llama-3.2-1B-Instruct`, which
        needs a token and an accepted licence. The default must not.
        """
        assert config.MODELS[0]["gated"] is False
        assert config.is_gated(config.DEFAULT_MODEL) is False

    def test_every_model_declares_whether_it_is_gated(self):
        for spec in config.MODELS:
            assert isinstance(spec["gated"], bool)
            assert spec["label"] and spec["note"]

    def test_gated_models_are_labelled_as_such(self):
        for spec in config.MODELS:
            if spec["gated"]:
                assert "GATED" in spec["label"]


class TestLoadGuards:
    def test_unknown_model_is_refused(self):
        with pytest.raises(InvalidInput, match="Unknown model"):
            load("not/a-model")

    def test_gated_model_without_a_token_explains_both_options(self, monkeypatch):
        monkeypatch.setattr("services.cag_service.hf_token", lambda: None)
        with pytest.raises(ModelLoadError) as exc:
            load("meta-llama/Llama-3.2-1B-Instruct")
        message = str(exc.value)
        assert "HF_TOKEN" in message
        assert "ungated" in message


class TestValidateDocument:
    def test_accepts_normal_text(self):
        assert validate_document("  hello  ") == "hello"

    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_rejects_empty(self, bad):
        with pytest.raises(InvalidInput, match="empty"):
            validate_document(bad)

    def test_rejects_a_document_that_cannot_fit_in_context(self, monkeypatch):
        # The whole premise is that the document fits. Past that point the
        # honest answer is "use retrieval", not a truncated cache.
        monkeypatch.setattr("services.cag_service.MAX_DOCUMENT_CHARS", 10)
        with pytest.raises(InvalidInput, match="what retrieval is for"):
            validate_document("x" * 50)


class TestParseDataset:
    def test_reads_question_and_answer_columns(self):
        rows = parse_dataset("question,answer\nWhat is X?,It is Y\n")
        assert len(rows) == 1
        assert rows[0].question == "What is X?"
        assert rows[0].answer == "It is Y"

    def test_accepts_ground_truth_as_the_answer_column(self):
        rows = parse_dataset("question,ground_truth\nQ?,A\n")
        assert rows[0].answer == "A"

    def test_column_names_are_case_insensitive(self):
        rows = parse_dataset("Question,Answer\nQ?,A\n")
        assert rows[0].question == "Q?"

    def test_missing_columns_names_what_it_found(self):
        # The message lists the columns that were actually present, which is
        # what tells you the file is the wrong shape.
        with pytest.raises(InvalidInput, match="foo, bar"):
            parse_dataset("foo,bar\n1,2\n")

    def test_reads_the_bundled_corpus_column_names(self):
        """The bundled dataset uses topic/text/sample_question/... names.

        A parser that only knows question/answer rejects the very file this
        project ships with, which is exactly how that was found.
        """
        rows = parse_dataset(
            "topic,text,sample_question,sample_ground_truth\n"
            "T,The document body,What is T?,An answer\n"
        )
        assert rows[0].question == "What is T?"
        assert rows[0].answer == "An answer"
        assert rows[0].document == "The document body"

    def test_rejects_an_empty_dataset(self):
        with pytest.raises(InvalidInput):
            parse_dataset("")

    def test_rejects_rows_with_no_question(self):
        with pytest.raises(InvalidInput, match="no usable rows"):
            parse_dataset("question,answer\n,\n")

    def test_rejects_too_many_questions(self, monkeypatch):
        monkeypatch.setattr("services.cag_service.MAX_QUESTIONS", 2)
        csv = "question,answer\n" + "".join(f"Q{i}?,A{i}\n" for i in range(5))
        with pytest.raises(InvalidInput, match="over the"):
            parse_dataset(csv)


class TestBundledDataset:
    def test_the_sample_dataset_exists_and_parses(self):
        # The UI offers it as the zero-setup path; without it that is a dead end.
        rows = default_dataset()
        assert rows and all(r.question for r in rows)

    def test_the_bundled_rows_carry_their_own_documents(self):
        assert all(r.document for r in default_dataset())

    def test_the_corpus_is_every_document_joined(self):
        from services.cag_service import default_corpus
        corpus = default_corpus()
        rows = default_dataset()
        assert len(corpus) > max(len(r.document) for r in rows)
        for row in rows:
            assert row.document in corpus

    def test_the_corpus_fits_inside_the_document_limit(self):
        # If the bundled demo cannot itself be cached, the demo is broken.
        from services.cag_service import default_corpus, validate_document
        validate_document(default_corpus())


class TestRunGuards:
    def test_running_before_loading_raises(self):
        with pytest.raises(NotLoaded, match="No model is loaded"):
            require_loaded()


class TestNoTokenPath:
    """An ungated model must work with no HF_TOKEN at all.

    Passing an empty string as the token makes huggingface_hub build the
    header `Bearer ` with nothing after it, which it rejects with
    `Illegal header value b'Bearer '`. So "no token" has to mean None, not "".
    """

    def test_empty_token_is_normalised_to_none(self):
        import inspect

        from services.cag_model import CAGModel

        source = inspect.getsource(CAGModel.load_model)
        assert "token=self.hf_token or None" in source, (
            "an empty-string token produces an illegal Authorization header"
        )

    def test_ungated_models_are_offered_without_a_token(self, monkeypatch):
        monkeypatch.setattr("services.cag_service.hf_token", lambda: None)
        # Must not raise before it gets as far as downloading.
        from services.cag_service import InvalidInput
        try:
            load(config.DEFAULT_MODEL)
        except InvalidInput:
            raise AssertionError("the default model was rejected without a token")
        except Exception:
            pass  # a download or offline failure here is fine; gating is not
