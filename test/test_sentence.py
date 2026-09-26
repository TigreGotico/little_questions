"""Unit tests for little_questions.Sentence and public API.

All tests that need classifiers mock them so the suite runs without models.
Tests that exercise real inference are marked with @pytest.mark.integration
and skipped unless --integration flag is passed.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_eat(label: str = "DESC:def") -> MagicMock:
    m = MagicMock()
    scores = {label: 0.9}
    m.score.return_value = scores
    m.predict.return_value = label
    return m


def _mock_sent(sent_type: str = "question") -> MagicMock:
    m = MagicMock()
    m.predict.return_value = sent_type
    m.score.return_value = {sent_type: 1.0}
    return m


def _make(text: str, sent_type: str = "question", eat_label: str = "DESC:def"):
    """Build a Sentence with mocked classifiers."""
    from little_questions import Sentence
    with (
        patch("little_questions.EatClassifier.get_instance", return_value=_mock_eat(eat_label)),
        patch("little_questions.SentenceTypeClassifier.get_instance", return_value=_mock_sent(sent_type)),
    ):
        return Sentence(text)


# ---------------------------------------------------------------------------
# Subclass dispatch
# ---------------------------------------------------------------------------

class TestSentenceSubclass:
    def test_question(self):
        from little_questions import Question
        s = _make("What is this?", "question")
        assert isinstance(s, Question)

    def test_statement(self):
        from little_questions import Statement
        s = _make("The sky is blue.", "statement")
        assert isinstance(s, Statement)

    def test_command(self):
        from little_questions import Command
        s = _make("Open the door.", "command")
        assert isinstance(s, Command)

    def test_exclamation(self):
        from little_questions import Exclamation
        s = _make("What a day!", "exclamation")
        assert isinstance(s, Exclamation)

    def test_request(self):
        from little_questions import Request, Command
        s = _make("Could you help?", "request")
        assert isinstance(s, Request)
        assert isinstance(s, Command)   # Request subclasses Command

    def test_unknown_type_is_plain_sentence(self):
        from little_questions import Sentence
        s = _make("Hmm.", "unknown_type")
        assert type(s) is Sentence

    def test_parse_classmethod_same_as_constructor(self):
        from little_questions import Sentence
        with (
            patch("little_questions.EatClassifier.get_instance", return_value=_mock_eat()),
            patch("little_questions.SentenceTypeClassifier.get_instance", return_value=_mock_sent()),
        ):
            s1 = Sentence.parse("Who are you?")
            s2 = Sentence("Who are you?")
        assert type(s1) is type(s2)
        assert s1.classification == s2.classification


# ---------------------------------------------------------------------------
# is_* boolean properties
# ---------------------------------------------------------------------------

class TestIsProperties:
    def test_is_question(self):
        s = _make("Who are you?", "question")
        assert s.is_question and not s.is_command and not s.is_statement

    def test_is_command(self):
        s = _make("Close the door.", "command")
        assert s.is_command and not s.is_question

    def test_is_request_implies_is_command(self):
        s = _make("Would you mind?", "request")
        assert s.is_request and s.is_command

    def test_is_statement(self):
        s = _make("Paris is in France.", "statement")
        assert s.is_statement

    def test_is_exclamation(self):
        s = _make("How wonderful!", "exclamation")
        assert s.is_exclamation


# ---------------------------------------------------------------------------
# EAT label properties
# ---------------------------------------------------------------------------

class TestEATLabels:
    @pytest.mark.parametrize("label,main,sec", [
        ("HUM:ind",  "HUM",  "ind"),
        ("ENTY:body","ENTY", "body"),
        ("DESC:def", "DESC", "def"),
        ("NUM:dist",  "NUM",  "dist"),
        ("LOC:city",  "LOC",  "city"),
        ("ABBR:abb",  "ABBR", "abb"),
        ("BOOL:yesno","BOOL", "yesno"),
    ])
    def test_main_and_secondary(self, label, main, sec):
        s = _make("x", eat_label=label)
        assert s.main_label == main
        assert s.secondary_label == sec

    def test_pretty_label_hum_ind(self):
        s = _make("Who is Einstein?", eat_label="HUM:ind")
        assert "Human" in s.pretty_label
        assert "individual" in s.pretty_label

    def test_pretty_label_bool_yesno(self):
        s = _make("Is it raining?", eat_label="BOOL:yesno")
        assert "Boolean" in s.pretty_label
        assert "yes/no" in s.pretty_label

    def test_pretty_label_desc_def(self):
        s = _make("What is gravity?", eat_label="DESC:def")
        assert "Description" in s.pretty_label

    def test_confidence_matches_max_score(self):
        from little_questions import Sentence
        eat_mock = MagicMock()
        eat_mock.score.return_value = {"HUM:ind": 0.7, "DESC:def": 0.2, "ENTY:other": 0.1}
        with (
            patch("little_questions.EatClassifier.get_instance", return_value=eat_mock),
            patch("little_questions.SentenceTypeClassifier.get_instance", return_value=_mock_sent()),
        ):
            s = Sentence("Who invented the telephone?")
        assert s.classification == "HUM:ind"
        assert s.confidence == pytest.approx(0.7)

    def test_classification_scores_keys(self):
        from little_questions import Sentence
        eat_mock = MagicMock()
        labels = {"HUM:ind": 0.6, "HUM:gr": 0.3, "DESC:def": 0.1}
        eat_mock.score.return_value = labels
        with (
            patch("little_questions.EatClassifier.get_instance", return_value=eat_mock),
            patch("little_questions.SentenceTypeClassifier.get_instance", return_value=_mock_sent()),
        ):
            s = Sentence("test")
        assert set(s.classification_scores.keys()) == set(labels.keys())


# ---------------------------------------------------------------------------
# Sentence still behaves like a str
# ---------------------------------------------------------------------------

class TestStrBehaviour:
    def test_upper(self):
        s = _make("hello world", "statement")
        assert s.upper() == "HELLO WORLD"

    def test_len(self):
        s = _make("hello", "statement")
        assert len(s) == 5

    def test_contains(self):
        s = _make("open sesame", "command")
        assert "sesame" in s

    def test_str_identity(self):
        s = _make("test string", "statement")
        assert str(s) == "test string"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

class TestPublicHelpers:
    def test_get_classifier_returns_eat_instance(self):
        from little_questions import get_classifier
        from little_questions.classifiers import EatClassifier
        clf = get_classifier("en")
        assert isinstance(clf, EatClassifier)

    def test_get_scorer_returns_sentence_type_instance(self):
        from little_questions import get_scorer
        from little_questions.classifiers import SentenceTypeClassifier
        scorer = get_scorer("en")
        assert isinstance(scorer, SentenceTypeClassifier)

    def test_clear_classifier_cache(self):
        from little_questions.classifiers import clear_classifier_cache, EatClassifier
        EatClassifier._instances["test_lang"] = MagicMock()
        clear_classifier_cache()
        assert "test_lang" not in EatClassifier._instances

    def test_list_supported_languages(self):
        from little_questions.classifiers import list_supported_languages
        langs = list_supported_languages()
        assert isinstance(langs, list)
        assert "en" in langs
        assert len(langs) >= 7


# ---------------------------------------------------------------------------
# Statement.answer_polarity (mocked yes/no classifier)
# ---------------------------------------------------------------------------

def _mock_yesno(polarity: str = "yes") -> MagicMock:
    m = MagicMock()
    scores = {"yes": 0.0, "no": 0.0, "maybe": 0.0}
    scores[polarity] = 0.9
    m.predict.return_value = polarity
    m.score.return_value = scores
    return m


def _make_statement(text: str, polarity: str = "yes"):
    from little_questions import Sentence
    with (
        patch("little_questions.EatClassifier.get_instance", return_value=_mock_eat("DESC:def")),
        patch("little_questions.SentenceTypeClassifier.get_instance", return_value=_mock_sent("statement")),
    ):
        s = Sentence(text)
    return s, polarity


class TestAnswerPolarity:
    def test_is_affirmative(self):
        from little_questions import Statement
        s, _ = _make_statement("Yes, it is.")
        with patch("little_questions.YesNoClassifier.get_instance", return_value=_mock_yesno("yes")):
            assert isinstance(s, Statement)
            assert s.answer_polarity == "yes"
            assert s.is_affirmative
            assert not s.is_negative

    def test_is_negative(self):
        from little_questions import Statement
        s, _ = _make_statement("No, it is not.")
        with patch("little_questions.YesNoClassifier.get_instance", return_value=_mock_yesno("no")):
            assert isinstance(s, Statement)
            assert s.answer_polarity == "no"
            assert s.is_negative
            assert not s.is_affirmative

    def test_maybe(self):
        from little_questions import Statement
        s, _ = _make_statement("Maybe, it depends.")
        with patch("little_questions.YesNoClassifier.get_instance", return_value=_mock_yesno("maybe")):
            assert isinstance(s, Statement)
            assert s.answer_polarity == "maybe"
            assert not s.is_affirmative
            assert not s.is_negative

    def test_polarity_scores_sum_to_one(self):
        from little_questions import Statement
        s, _ = _make_statement("Yes indeed.")
        mock = _mock_yesno("yes")
        mock.score.return_value = {"yes": 0.85, "no": 0.10, "maybe": 0.05}
        with patch("little_questions.YesNoClassifier.get_instance", return_value=mock):
            scores = s.answer_polarity_scores
            assert set(scores.keys()) == {"yes", "no", "maybe"}
            assert sum(scores.values()) == pytest.approx(1.0, abs=0.01)

    def test_no_model_raises(self):
        from little_questions import Statement
        s, _ = _make_statement("Sure.")
        with patch("little_questions.YesNoClassifier.get_instance",
                   side_effect=RuntimeError("no model")):
            with pytest.raises(RuntimeError):
                _ = s.answer_polarity

    def test_polarity_cached(self):
        """answer_polarity should only call the classifier once."""
        from little_questions import Statement
        s, _ = _make_statement("Yes.")
        mock = _mock_yesno("yes")
        with patch("little_questions.YesNoClassifier.get_instance", return_value=mock):
            _ = s.answer_polarity
            _ = s.answer_polarity
        mock.predict.assert_called_once()


# ---------------------------------------------------------------------------
# PunctuationScorer baseline (train module)
# ---------------------------------------------------------------------------

class TestPunctuationScorer:
    def test_question_mark(self):
        from train.baselines import PunctuationScorer
        scores = PunctuationScorer().score("Is this a test?")
        assert scores["question"] > scores["statement"]

    def test_exclamation_mark(self):
        from train.baselines import PunctuationScorer
        scores = PunctuationScorer().score("What a surprise!")
        assert scores["exclamation"] > scores["statement"]

    def test_predict_returns_valid_type(self):
        from train.baselines import PunctuationScorer
        result = PunctuationScorer().predict("What is this?")
        assert result in {"question", "statement", "exclamation", "command", "request"}


# ---------------------------------------------------------------------------
# Integration tests (require real models — skip by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    def test_who_question_classified_as_hum(self):
        from little_questions import Sentence, Question
        s = Sentence("Who invented the telephone?")
        assert isinstance(s, Question)
        assert s.main_label == "HUM"

    def test_what_is_question_classified_as_desc(self):
        from little_questions import Sentence, Question
        s = Sentence("What is photosynthesis?")
        assert isinstance(s, Question)
        assert s.main_label in ("DESC", "ENTY")

    def test_confidence_in_range(self):
        from little_questions import Sentence
        s = Sentence("Where is Paris?")
        assert 0.0 < s.confidence <= 1.0

    def test_scores_sum_to_one(self):
        from little_questions import Sentence
        s = Sentence("When did World War II end?")
        total = sum(s.classification_scores.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_bool_yesno_classification(self):
        from little_questions import Sentence
        s = Sentence("Is the Eiffel Tower in Paris?")
        assert s.main_label == "BOOL"

    def test_statement_type(self):
        from little_questions import Sentence, Statement
        s = Sentence("The sky is blue.")
        assert isinstance(s, Statement)
