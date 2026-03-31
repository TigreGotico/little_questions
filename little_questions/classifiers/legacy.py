"""
Legacy heuristic-based sentence type classifier using POS tagging.

This module is kept as a fallback for non-English languages or when
the trained classifier is unavailable. For English, use the trained
SentenceTypeClassifier in little_questions.sentence_type instead.
"""

from typing import Dict, List, Optional

from nltk import word_tokenize, pos_tag


# begin of sentence indicators for Yes/No questions
YES_NO_STARTERS: frozenset = frozenset({
    "would", "is", "will", "does", "can", "has", "if",
    "could", "are", "should", "have", "did",
})

# begin of sentence indicators for "command" questions, eg, "do this"
# non exhaustive list, should capture common voice interactions
COMMAND_STARTERS = ["name", "define", "list", "tell", "say"]

ALL_POS_TAGS = [
    "NNPS",
    "--",
    ".",
    "POS",
    "RB",
    "UH",
    "SYM",
    "(",
    "JJR",
    "WDT",
    "PRP",
    "NNS",
    "JJS",
    "$",
    "JJ",
    "IN",
    "EX",
    "CC",
    "NN",
    "MD",
    "``",
    ",",
    "RBR",
    ":",
    "PDT",
    "WP",
    "RP",
    "WP$",
    "TO",
    "VBP",
    "WRB",
    "VB",
    "VBG",
    "VBN",
    ")",
    "DT",
    "''",
    "PRP$",
    "VBZ",
    "VBD",
    "FW",
    "LS",
    "CD",
    "NNP",
    "RBS",
]


class SentenceScorerHeuristic:
    """Legacy heuristic-based sentence type scorer using POS tagging.

    This class is kept as a fallback for non-English languages or when
    the trained classifier is unavailable. For English, use the trained
    SentenceTypeClassifier instead.
    """

    @staticmethod
    def predict(text: str) -> str:
        score = SentenceScorerHeuristic.score(text)
        best = max(score, key=lambda key: score[key])
        return best

    @staticmethod
    def score(text: str) -> Dict[str, float]:
        return {
            "question": SentenceScorerHeuristic.question_score(text),
            "statement": SentenceScorerHeuristic.statement_score(text),
            "exclamation": SentenceScorerHeuristic.exclamation_score(text),
            "command": SentenceScorerHeuristic.command_score(text),
            "request": SentenceScorerHeuristic.request_score(text),
        }

    @staticmethod
    def _score(
        text: str,
        last_tokens: Optional[List[str]] = None,
        first_tokens: Optional[List[str]] = None,
        start_pos_tags: Optional[List[str]] = None,
        end_pos_tags: Optional[List[str]] = None,
        unlikely_words: Optional[List[str]] = None,
        unlikely_start_pos_tag: Optional[List[str]] = None,
        unlikely_end_pos_tag: Optional[List[str]] = None,
        unlikely_pos_tag: Optional[List[str]] = None,
    ) -> float:
        score = 8
        last_tokens = last_tokens or []
        first_tokens = first_tokens or []
        start_pos_tags = start_pos_tags or []
        end_pos_tags = end_pos_tags or []
        unlikely_words = unlikely_words or []
        unlikely_start_pos_tag = unlikely_start_pos_tag or [
            t for t in ALL_POS_TAGS if t not in start_pos_tags
        ]
        unlikely_end_pos_tag = unlikely_end_pos_tag or []
        unlikely_pos_tag = (
            unlikely_pos_tag or unlikely_end_pos_tag + unlikely_start_pos_tag
        )
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
        for pos in tagged:
            if pos[1] in unlikely_pos_tag:
                score -= 0.1
        for tok in tokens:
            if tok in unlikely_words:
                score -= 0.2
        if score <= 0:
            return 0
        return max(score / 7, 0)

    @staticmethod
    def question_score(text: str) -> float:
        """Questions can have two patterns. Some can have 'yes' or 'no' as an answer."""
        last_tokens = ["?"]
        first_tokens = [
            "what",
            "why",
            "how",
            "when",
            "who",
            "whose",
            "which",
        ] + YES_NO_STARTERS
        start_pos_tags = ["WDT", "WP", "WP$", "WRB", "VB", "VBP", "VBZ", "VBN", "VBG"]
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"]
        unlikely_words = ["!"]
        unlikely_start_pos_tag = [t for t in ALL_POS_TAGS if t not in start_pos_tags]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = []
        score = SentenceScorerHeuristic._score(
            text,
            last_tokens,
            first_tokens,
            start_pos_tags,
            end_pos_tags,
            unlikely_words,
            unlikely_start_pos_tag,
            unlikely_end_pos_tag,
            unlikely_pos_tag,
        )

        starts = ["in what ", "on what ", "at what ", "in which"]
        for s in starts:
            if text.lower().startswith(s):
                score += 0.1
                break
        ends = [" in what", " on what", " for what", " as what", "?"]
        for s in ends:
            if text.lower().endswith(s):
                score += 0.1
                break
        return min(score, 1)

    @staticmethod
    def statement_score(text: str) -> float:
        """A statement is defined as having a structure with Subject + Verb + Object."""
        last_tokens = ["."]
        first_tokens = ["we", "i", "the", "our"]
        start_pos_tags = ["PRP", "PRP$", "DT"]
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"]
        unlikely_words = [
            "what",
            "why",
            "how",
            "when",
            "who",
            "whose",
            "which",
            "?",
            "!",
        ]
        unlikely_start_pos_tag = [
            "WDT",
            "WP",
            "WP$",
            "WRB",
            "VB",
            "VBP",
            "VBZ",
            "VBN",
            "VBG",
        ]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = ["WDT", "WP", "WP$", "WRB"]
        score = SentenceScorerHeuristic._score(
            text,
            last_tokens,
            first_tokens,
            start_pos_tags,
            end_pos_tags,
            unlikely_words,
            unlikely_start_pos_tag,
            unlikely_end_pos_tag,
            unlikely_pos_tag,
        )

        if text.lower().startswith("be "):
            score -= 0.1
        return max(min(score + 0.0001, 1), 0)

    @staticmethod
    def exclamation_score(text: str) -> float:
        """Exclamations grammatically have a structure involving 'what a' or 'how'."""
        last_tokens = ["!"]
        first_tokens = ["how", "what"]
        start_pos_tags = ["WP", "WRB"]
        end_pos_tags = [
            ".",
            "NN",
            "NNP",
            "NNS",
            "NNPS",
            "PRP",
            "JJ",
            "VB",
            "VBP",
            "VBZ",
            "VBN",
            "VBG",
        ]
        unlikely_words = ["why", "when", "who", "whose", "which", "?"]
        unlikely_start_pos_tag = [
            "WDT",
            "WP$",
            "NN",
            "NNP",
            "NNS",
            "NNPS",
            "DT",
            "PRP",
            "JJ",
            "VB",
            "VBP",
            "VBZ",
            "VBN",
            "VBG",
            "PDT",
            "RB",
            "RBR",
            "RBS",
        ]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = ["WDT", "WP$"]
        score = SentenceScorerHeuristic._score(
            text,
            last_tokens,
            first_tokens,
            start_pos_tags,
            end_pos_tags,
            unlikely_words,
            unlikely_start_pos_tag,
            unlikely_end_pos_tag,
            unlikely_pos_tag,
        )

        if text.split(" ")[0].lower() not in first_tokens:
            score -= 0.1
        elif not (text.lower().startswith("what a ") or text.lower().startswith("what an ")):
            score -= 0.1
        for w in unlikely_words:
            if w in text.lower():
                score -= 0.05

        common_mistakes = [
            "how many",
            "how much",
            "how tall",
            "how fast",
            "how big",
            "how often",
            "what is",
            "what are",
        ]
        for w in common_mistakes:
            if text.lower().startswith(w):
                score -= 0.1
        return max(score, 0)

    @staticmethod
    def command_score(text: str) -> float:
        """Commands typically lack a Subject and start with infinitive verb."""
        last_tokens = ["!", "."]
        first_tokens = ["be"]
        start_pos_tags = ["VB", "NN", "NNP"]
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP"]
        unlikely_words = ["what", "why", "how", "when", "who", "whose", "which", "?"]
        unlikely_start_pos_tag = ["WDT", "WP", "WP$", "WRB", "JJ"]
        unlikely_end_pos_tag = ["VB", "VBP", "VBZ", "VBN", "VBG"]
        unlikely_pos_tag = ["WDT", "WP", "WP$", "WRB"]
        score = SentenceScorerHeuristic._score(
            text,
            last_tokens,
            first_tokens,
            start_pos_tags,
            end_pos_tags,
            unlikely_words,
            unlikely_start_pos_tag,
            unlikely_end_pos_tag,
            unlikely_pos_tag,
        )

        starts = ["do the ", "do your ", "name", "define"]
        for s in starts:
            if text.lower().startswith(s):
                score += 0.1
                break
        return min(score, 1)

    @staticmethod
    def request_score(text: str) -> float:
        """Requests use polite modal verbs like would/could/can."""
        last_tokens = [".", "?"]
        first_tokens = ["would", "could", "can"]
        start_pos_tags = ["MD"]
        end_pos_tags = [".", "NN", "NNP", "NNS", "NNPS", "PRP", "JJ"]
        unlikely_words = ["!", "?"]
        unlikely_start_pos_tag = [t for t in ALL_POS_TAGS if t not in start_pos_tags]
        unlikely_end_pos_tag = []
        unlikely_pos_tag = []
        score = SentenceScorerHeuristic._score(
            text,
            last_tokens,
            first_tokens,
            start_pos_tags,
            end_pos_tags,
            unlikely_words,
            unlikely_start_pos_tag,
            unlikely_end_pos_tag,
            unlikely_pos_tag,
        )

        if "please" in text:
            score += 0.3
        if text.split(" ")[0].lower() not in first_tokens:
            score -= 0.2
        return max(min(score, 1), 0)
