"""Rule-based sentence-type baseline scorers for benchmarking.

These scorers are intentionally NOT used in production inference. They exist
to provide a reproducible baseline when evaluating the trained classifiers.

Usage::

    from train.baselines import PunctuationScorer, HeuristicScorer
    from train.metrics import evaluate_scorer

    scorer = HeuristicScorer()
    report = evaluate_scorer(scorer, texts, labels)
    print(report.classification_report())
"""

from __future__ import annotations

from typing import Dict

from nltk import word_tokenize, pos_tag


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YES_NO_STARTERS: frozenset = frozenset({
    "would", "is", "will", "does", "can", "has", "if",
    "could", "are", "should", "have", "did",
})

COMMAND_STARTERS = ["name", "define", "list", "tell", "say"]

ALL_POS_TAGS = frozenset({
    "NNPS", "--", ".", "POS", "RB", "UH", "SYM", "(", "JJR", "WDT",
    "PRP", "NNS", "JJS", "$", "JJ", "IN", "EX", "CC", "NN", "MD",
    "``", ",", "RBR", ":", "PDT", "WP", "RP", "WP$", "TO", "VBP",
    "WRB", "VB", "VBG", "VBN", ")", "DT", "''", "PRP$", "VBZ",
    "VBD", "FW", "LS", "CD", "NNP", "RBS",
})


# ---------------------------------------------------------------------------
# Punctuation baseline (language-agnostic)
# ---------------------------------------------------------------------------

class PunctuationScorer:
    """Baseline scorer that uses only terminal punctuation.

    Accuracy is low (~50-60%) but requires no models, no NLTK data,
    and works for any language.
    """

    name = "punctuation"

    def predict(self, text: str) -> str:
        """Return the most likely sentence type for *text*."""
        scores = self.score(text)
        return max(scores, key=lambda k: scores[k])

    def score(self, text: str) -> Dict[str, float]:
        """Return a confidence dict for each sentence type."""
        return {
            "question":    0.8 if text.endswith("?") else 0.4,
            "statement":   0.5 if text.endswith(".") else 0.0,
            "exclamation": 0.6 if text.endswith("!") else 0.0,
            "command":     (0.6 if text.endswith(".") else (0.5 if text.endswith("!") else 0.0)),
            "request":     (0.5 if text.endswith((".", "?")) else 0.0),
        }


# ---------------------------------------------------------------------------
# POS-tag heuristic baseline (English only)
# ---------------------------------------------------------------------------

class HeuristicScorer:
    """English-only baseline scorer using NLTK POS tagging.

    More accurate than PunctuationScorer (~70-75% on English sentence-type)
    but requires NLTK data (``punkt``, ``averaged_perceptron_tagger``) and
    is significantly slower (POS tagging per sentence).
    """

    name = "heuristic_pos"

    def predict(self, text: str) -> str:
        """Return the most likely sentence type for *text*."""
        scores = self.score(text)
        return max(scores, key=lambda k: scores[k])

    def score(self, text: str) -> Dict[str, float]:
        """Return a confidence dict for each sentence type."""
        return {
            "question":    self.question_score(text),
            "statement":   self.statement_score(text),
            "exclamation": self.exclamation_score(text),
            "command":     self.command_score(text),
            "request":     self.request_score(text),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(
        text: str,
        last_tokens: list[str],
        first_tokens: list[str],
        start_pos_tags: list[str],
        end_pos_tags: list[str],
        unlikely_words: list[str],
        unlikely_start_pos_tag: list[str] | None = None,
        unlikely_end_pos_tag: list[str] | None = None,
        unlikely_pos_tag: list[str] | None = None,
    ) -> float:
        score = 8
        unlikely_start_pos_tag = unlikely_start_pos_tag or [
            t for t in ALL_POS_TAGS if t not in start_pos_tags
        ]
        unlikely_end_pos_tag = unlikely_end_pos_tag or []
        unlikely_pos_tag = unlikely_pos_tag or (unlikely_end_pos_tag + unlikely_start_pos_tag)

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        if tokens[-1] not in last_tokens:
            score -= 1
        if tokens[0] not in first_tokens:
            score -= 1
        if tagged[0][1] not in start_pos_tags:
            score -= 1
        if tagged[-1][1] not in end_pos_tags:
            score -= 1
        if tagged[0][1] in unlikely_start_pos_tag:
            score -= 1
        if tagged[-1][1] in unlikely_end_pos_tag:
            score -= 1
        for _, pos in tagged:
            if pos in unlikely_pos_tag:
                score -= 0.1
        for tok in tokens:
            if tok in unlikely_words:
                score -= 0.2
        return max(score / 7, 0)

    def question_score(self, text: str) -> float:
        """Confidence that *text* is a question."""
        start_pos_tags = ["WDT", "WP", "WP$", "WRB", "VB", "VBP", "VBZ", "VBN", "VBG"]
        score = self._score(
            text,
            last_tokens=["?"],
            first_tokens=["what", "why", "how", "when", "who", "whose", "which"] + list(YES_NO_STARTERS),
            start_pos_tags=start_pos_tags,
            end_pos_tags=[".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"],
            unlikely_words=["!"],
            unlikely_start_pos_tag=[t for t in ALL_POS_TAGS if t not in start_pos_tags],
        )
        text_lower = text.lower()
        for prefix in ("in what ", "on what ", "at what ", "in which"):
            if text_lower.startswith(prefix):
                score += 0.1
                break
        for suffix in (" in what", " on what", " for what", " as what", "?"):
            if text_lower.endswith(suffix):
                score += 0.1
                break
        return min(score, 1)

    def statement_score(self, text: str) -> float:
        """Confidence that *text* is a statement."""
        score = self._score(
            text,
            last_tokens=["."],
            first_tokens=["we", "i", "the", "our"],
            start_pos_tags=["PRP", "PRP$", "DT"],
            end_pos_tags=[".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"],
            unlikely_words=["what", "why", "how", "when", "who", "whose", "which", "?", "!"],
            unlikely_start_pos_tag=["WDT", "WP", "WP$", "WRB", "VB", "VBP", "VBZ", "VBN", "VBG"],
            unlikely_pos_tag=["WDT", "WP", "WP$", "WRB"],
        )
        if text.lower().startswith("be "):
            score -= 0.1
        return max(min(score + 0.0001, 1), 0)

    def exclamation_score(self, text: str) -> float:
        """Confidence that *text* is an exclamation."""
        first_tokens = ["how", "what"]
        score = self._score(
            text,
            last_tokens=["!"],
            first_tokens=first_tokens,
            start_pos_tags=["WP", "WRB"],
            end_pos_tags=[".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ", "VB", "VBP", "VBZ", "VBN", "VBG"],
            unlikely_words=["why", "when", "who", "whose", "which", "?"],
            unlikely_start_pos_tag=["WDT", "WP$", "NN", "NNP", "NNS", "NNPS", "DT", "PRP", "JJ",
                                    "VB", "VBP", "VBZ", "VBN", "VBG", "PDT", "RB", "RBR", "RBS"],
            unlikely_pos_tag=["WDT", "WP$"],
        )
        text_lower = text.lower()
        if text_lower.split(" ")[0] not in first_tokens:
            score -= 0.1
        elif not (text_lower.startswith("what a ") or text_lower.startswith("what an ")):
            score -= 0.1
        unlikely_words = ["why", "when", "who", "whose", "which", "?"]
        for w in unlikely_words:
            if w in text_lower:
                score -= 0.05
        for phrase in ("how many", "how much", "how tall", "how fast", "how big",
                       "how often", "what is", "what are"):
            if text_lower.startswith(phrase):
                score -= 0.1
        return max(score, 0)

    def command_score(self, text: str) -> float:
        """Confidence that *text* is a command."""
        score = self._score(
            text,
            last_tokens=["!", "."],
            first_tokens=["be"],
            start_pos_tags=["VB", "NN", "NNP"],
            end_pos_tags=[".", "NN", "NNP", "NNS", "NNPS", "PRP"],
            unlikely_words=["what", "why", "how", "when", "who", "whose", "which", "?"],
            unlikely_start_pos_tag=["WDT", "WP", "WP$", "WRB", "JJ"],
            unlikely_end_pos_tag=["VB", "VBP", "VBZ", "VBN", "VBG"],
            unlikely_pos_tag=["WDT", "WP", "WP$", "WRB"],
        )
        for prefix in ("do the ", "do your ", "name", "define"):
            if text.lower().startswith(prefix):
                score += 0.1
                break
        return min(score, 1)

    def request_score(self, text: str) -> float:
        """Confidence that *text* is a request."""
        first_tokens = ["would", "could", "can"]
        score = self._score(
            text,
            last_tokens=[".", "?"],
            first_tokens=first_tokens,
            start_pos_tags=["MD"],
            end_pos_tags=[".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"],
            unlikely_words=["!", "?"],
            unlikely_start_pos_tag=[t for t in ALL_POS_TAGS if t != "MD"],
        )
        if "please" in text:
            score += 0.3
        if text.split(" ")[0].lower() not in first_tokens:
            score -= 0.2
        return max(min(score, 1), 0)
