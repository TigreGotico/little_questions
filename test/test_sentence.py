"""Unit tests for little_questions sentence classification.

Tests cover:
- Sentence type detection via SentenceScorer heuristics (mocked classifier)
- COSC label properties (main_label, secondary_label, pretty_label)
- is_* boolean properties
- Correct concrete subclass returned by Sentence.__new__
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sentence(text: str, sent_type: str, cosc_label: str = "DESC:def"):
    """Build a Sentence instance with mocked classifiers."""
    from little_questions import Sentence

    mock_qclf = MagicMock()
    mock_qclf.predict.return_value = [cosc_label]

    mock_scorer = MagicMock()
    mock_scorer.predict.return_value = sent_type
    mock_scorer.score.return_value = {
        "question": 0.0,
        "statement": 0.0,
        "exclamation": 0.0,
        "command": 0.0,
        "request": 0.0,
        sent_type: 1.0,
    }

    with (
        patch("little_questions.get_classifier", return_value=mock_qclf),
        patch("little_questions.get_scorer", return_value=mock_scorer),
    ):
        return Sentence(text)


# ---------------------------------------------------------------------------
# Sentence type classification — correct subclass
# ---------------------------------------------------------------------------


class TestSentenceSubclass:
    def test_question_returns_question(self):
        from little_questions import Question

        s = _make_sentence("What is the capital of France?", "question")
        assert isinstance(s, Question)

    def test_command_returns_command(self):
        from little_questions import Command

        s = _make_sentence("Open the door.", "command")
        assert isinstance(s, Command)

    def test_statement_returns_statement(self):
        from little_questions import Statement

        s = _make_sentence("The sky is blue.", "statement")
        assert isinstance(s, Statement)

    def test_exclamation_returns_exclamation(self):
        from little_questions import Exclamation

        s = _make_sentence("What a beautiful day!", "exclamation")
        assert isinstance(s, Exclamation)

    def test_request_returns_request(self):
        from little_questions import Request

        s = _make_sentence("Could you pass the salt?", "request")
        assert isinstance(s, Request)

    def test_unknown_type_returns_sentence(self):
        from little_questions import Sentence

        s = _make_sentence("Hmm.", "unknown_type")
        assert type(s) is Sentence


# ---------------------------------------------------------------------------
# is_* boolean properties
# ---------------------------------------------------------------------------


class TestIsBooleanProperties:
    def test_is_question(self):
        s = _make_sentence("Who are you?", "question")
        assert s.is_question is True
        assert s.is_command is False
        assert s.is_statement is False

    def test_is_command(self):
        s = _make_sentence("Shut the door.", "command")
        assert s.is_command is True
        assert s.is_question is False

    def test_is_request_also_is_command(self):
        from little_questions import Command

        s = _make_sentence("Would you mind?", "request")
        assert s.is_request is True
        # Request subclasses Command
        assert isinstance(s, Command)
        assert s.is_command is True

    def test_is_statement(self):
        s = _make_sentence("Paris is in France.", "statement")
        assert s.is_statement is True

    def test_is_exclamation(self):
        s = _make_sentence("How wonderful!", "exclamation")
        assert s.is_exclamation is True


# ---------------------------------------------------------------------------
# COSC label properties
# ---------------------------------------------------------------------------


class TestCOSCLabels:
    @pytest.mark.parametrize(
        "label,expected_main,expected_sec",
        [
            ("HUM:ind", "HUM", "ind"),
            ("ENTY:body", "ENTY", "body"),
            ("DESC:def", "DESC", "def"),
            ("NUM:dist", "NUM", "dist"),
            ("LOC:city", "LOC", "city"),
            ("ABBR:abb", "ABBR", "abb"),
        ],
    )
    def test_main_and_secondary_labels(self, label, expected_main, expected_sec):
        s = _make_sentence("Who invented the telephone?", "question", label)
        assert s.main_label == expected_main
        assert s.secondary_label == expected_sec

    def test_pretty_label_human_individual(self):
        s = _make_sentence("Who is Einstein?", "question", "HUM:ind")
        pretty = s.pretty_label
        assert "Human" in pretty
        assert "individual" in pretty

    def test_pretty_label_entity_body(self):
        s = _make_sentence("What organ pumps blood?", "question", "ENTY:body")
        pretty = s.pretty_label
        assert "Entity" in pretty
        assert "organs of body" in pretty

    def test_pretty_label_desc_def(self):
        s = _make_sentence("What is photosynthesis?", "question", "DESC:def")
        pretty = s.pretty_label
        assert "Description" in pretty
        assert "definition" in pretty

    def test_pretty_label_abbr_abb(self):
        s = _make_sentence("What does NASA stand for?", "question", "ABBR:abb")
        pretty = s.pretty_label
        assert "Abbreviation" in pretty
        assert "abbreviation" in pretty


# ---------------------------------------------------------------------------
# Sentence still behaves like a str
# ---------------------------------------------------------------------------


class TestStrBehaviour:
    def test_upper(self):
        s = _make_sentence("hello world", "statement")
        # upper() on str subclass returns a plain str (no re-wrapping needed)
        assert s.upper() == "HELLO WORLD"

    def test_len(self):
        s = _make_sentence("hello", "statement")
        assert len(s) == 5

    def test_contains(self):
        s = _make_sentence("open sesame", "command")
        assert "sesame" in s


# ---------------------------------------------------------------------------
# SentenceScorer heuristics (no model required)
# ---------------------------------------------------------------------------


class TestPunctuationBaselineScorer:
    """Tests for the baseline punctuation scorer (now in train/baselines.py)."""

    def test_question_mark_scores_highest_for_question(self):
        from train.baselines import PunctuationScorer

        scores = PunctuationScorer().score("Is this a test?")
        assert scores["question"] > scores["statement"]
        assert scores["question"] > scores["command"]

    def test_exclamation_mark_scores_highest_for_exclamation(self):
        from train.baselines import PunctuationScorer

        scores = PunctuationScorer().score("What a surprise!")
        assert scores["exclamation"] > scores["statement"]

    def test_period_scores_command(self):
        from train.baselines import PunctuationScorer

        scores = PunctuationScorer().score("Do this.")
        assert scores["command"] >= scores["question"]

    def test_predict_returns_string(self):
        from train.baselines import PunctuationScorer

        result = PunctuationScorer().predict("What is this?")
        assert isinstance(result, str)
        assert result in {"question", "statement", "exclamation", "command", "request"}


# ---------------------------------------------------------------------------
# Phase 2 — Clean API tests
# ---------------------------------------------------------------------------


class TestModelPathBug:
    """Regression test: LANG2MODEL entries must end with .onnx for all languages."""

    @pytest.mark.parametrize("lang", ["en", "es", "pt", "ca", "fr", "de", "it", "nl"])
    def test_get_model_path_returns_onnx_for_all_languages(self, lang):
        from little_questions.models import LANG2MODEL

        assert lang in LANG2MODEL, f"{lang} not in LANG2MODEL"
        filename = LANG2MODEL[lang]
        assert filename.endswith(".onnx"), (
            f"{lang}: filename {filename!r} missing .onnx extension"
        )


class TestSentenceParse:
    """Test Sentence.parse() classmethod."""

    def test_parse_returns_correct_subclass(self):
        from little_questions import Sentence, Question

        with (
            patch("little_questions.get_classifier") as mock_clf,
            patch("little_questions.get_scorer") as mock_scorer,
        ):
            mock_clf.return_value.predict.return_value = ["DESC:def"]
            mock_scorer.return_value.predict.return_value = "question"
            mock_scorer.return_value.score.return_value = {
                "question": 1.0,
                "statement": 0.0,
                "exclamation": 0.0,
                "command": 0.0,
                "request": 0.0,
            }
            s = Sentence.parse("What is the capital of France?", lang="en")
            assert isinstance(s, Question)

    def test_parse_same_as_constructor(self):
        from little_questions import Sentence

        with (
            patch("little_questions.get_classifier") as mock_clf,
            patch("little_questions.get_scorer") as mock_scorer,
        ):
            mock_clf.return_value.predict.return_value = ["DESC:def"]
            mock_scorer.return_value.predict.return_value = "question"
            mock_scorer.return_value.score.return_value = {
                "question": 1.0,
                "statement": 0.0,
                "exclamation": 0.0,
                "command": 0.0,
                "request": 0.0,
            }
            s1 = Sentence.parse("Who are you?", lang="en")
            s2 = Sentence("Who are you?", model="en")
            assert type(s1) is type(s2)
            assert s1.classification == s2.classification
            assert s1.sentence_type == s2.sentence_type


class TestClassifierCache:
    """Test classifier cache utilities."""

    def test_clear_classifier_cache(self):
        from little_questions.classifiers import _LAZY_LOADING, clear_classifier_cache

        _LAZY_LOADING["test_key"] = "dummy"
        assert "test_key" in _LAZY_LOADING
        clear_classifier_cache()
        assert "test_key" not in _LAZY_LOADING

    def test_list_supported_languages(self):
        from little_questions.classifiers import list_supported_languages

        langs = list_supported_languages()
        assert isinstance(langs, list)
        assert set(langs) == {"en", "es", "pt", "ca", "fr", "de", "it", "nl"}
