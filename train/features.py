"""Linguistic feature extraction for COSC and sentence-type classification.

Provides a sklearn-compatible transformer that extracts 40+ features from text:
POS tag counts, POS bigram transitions, lexical diversity, first-word intent
signals, sentence length, and binary intent flags.

Based on the feature engineering from
``guided-categorical-embeddings/examples/questions_experiment_1/process.py``.

Usage::

    from train.features import LinguisticFeaturesTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    pipe = Pipeline([
        ("feats", LinguisticFeaturesTransformer()),
        ("clf", LinearSVC()),
    ])
    pipe.fit(x_train, y_train)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

YES_NO_STARTERS = frozenset({
    "were", "are", "will", "do", "if", "have", "did", "has", "does", "can",
    "shall", "am", "should", "would", "was", "is", "might", "must", "could", "may",
})

QUESTION_STARTERS = frozenset({
    "where", "when", "how", "why", "who", "whose", "which", "what", "whom",
})

QUESTION_PHRASES = (
    "how many", "how much", "how long", "how often", "how far", "how tall",
    "how old", "how big",
)

COMMAND_STARTERS = (
    "don't forget ", "do not forget ", "please ", "make sure to ", "remember to ",
    "ensure you ", "do not forget to ", "don't forget to ", "make sure you ",
)

DENIAL_STARTERS = (
    "don't ", "do not ", "no ", "never ", "not at all ", "no way ",
    "under no circumstances ",
)

THANKS_WORDS = frozenset({
    "thanks", "thank you", "thanks a lot", "thanks very much",
    "many thanks", "thanks so much",
})

PLEASE_WORDS = frozenset({
    "please", "kindly", "would you mind", "if you could", "could you please",
})

DENIAL_WORDS = frozenset({
    "forbidden", "prohibited", "not permitted", "not allowed", "unacceptable",
    "restricted", "banned", "out of bounds",
})

PLAY_VERBS = frozenset({
    "play", "watch", "listen", "view", "launch", "open", "start", "activate",
})

SUBJECT_TAGS = frozenset({"NN", "NNP", "NNS", "NNPS", "PRP"})
VERB_TAGS = frozenset({"VB", "VBP", "VBZ", "VBD", "VBN", "VBG"})

POS_TAGS_OF_INTEREST = (
    "NN", "NNP", "NNS", "NNPS", "PRP", "PRP$",
    "VB", "VBP", "VBZ", "VBD", "VBN", "VBG",
    "JJ", "JJR", "JJS",
    "RB", "RBR", "RBS",
    "WP", "WP$", "WRB", "WDT",
    "DT", "IN", "CC", "MD",
)


def sentence_to_features(sentence: str) -> Dict[str, float]:
    """Extract 40+ linguistic features from a single sentence.

    Args:
        sentence: Raw input text.

    Returns:
        Dict mapping feature name → numeric value (bool features as 0/1 floats).
    """
    from nltk import word_tokenize, pos_tag

    s = sentence.lower().strip()
    tokens = word_tokenize(s)
    if not tokens:
        return {}

    tagged = pos_tag(tokens)
    pos_tags_list = [tag for _, tag in tagged]
    first_word = tokens[0]

    feats: Dict[str, float] = {}

    # --- Binary intent signals ---
    feats["is_yes_no"] = float(first_word in YES_NO_STARTERS)
    feats["is_wh"] = float(first_word in QUESTION_STARTERS)
    feats["is_wh_phrase"] = float(any(s.startswith(p) for p in QUESTION_PHRASES))
    feats["is_play"] = float(first_word in PLAY_VERBS)
    feats["is_cmd"] = float(any(s.startswith(p) for p in COMMAND_STARTERS))
    feats["is_denial_start"] = float(any(s.startswith(p) for p in DENIAL_STARTERS))
    feats["is_the"] = float(first_word == "the")
    feats["startswith_pron"] = float(first_word in {"he", "she", "it", "they", "their", "we", "i"})
    feats["ends_question_mark"] = float(sentence.strip().endswith("?"))
    feats["ends_exclamation"] = float(sentence.strip().endswith("!"))
    feats["ends_period"] = float(sentence.strip().endswith("."))

    # --- Lexical features ---
    feats["sentence_length"] = float(len(tokens))
    feats["avg_token_length"] = float(sum(len(t) for t in tokens) / len(tokens))
    feats["unique_token_count"] = float(len(set(tokens)))
    feats["lexical_diversity"] = float(len(set(tokens)) / len(tokens))
    feats["contains_conjunction"] = float(
        any(t in {"and", "or", "but", "so", "for", "nor", "yet"} for t in tokens)
    )
    feats["thanks_words"] = float(any(w in s for w in THANKS_WORDS))
    feats["please_words"] = float(any(w in s for w in PLEASE_WORDS))
    feats["denial_words"] = float(any(w in s for w in DENIAL_WORDS))

    # --- POS tag counts ---
    pos_counts: Dict[str, int] = {}
    for tag in pos_tags_list:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1
    for tag in POS_TAGS_OF_INTEREST:
        feats[f"pos_{tag}"] = float(pos_counts.get(tag, 0))

    # --- POS bigram transitions ---
    for i in range(len(pos_tags_list) - 1):
        key = f"trans_{pos_tags_list[i]}_{pos_tags_list[i + 1]}"
        feats[key] = feats.get(key, 0.0) + 1.0

    # --- Structural intent flags ---
    feats["is_statement"] = float(
        any(t in SUBJECT_TAGS for _, t in tagged[:2])
        and any(t in VERB_TAGS for _, t in tagged)
    )
    feats["is_exclamation_struct"] = float(
        (first_word == "what" and len(tokens) > 1 and tokens[1] in {"a", "an"})
        or (first_word == "how" and any(t == "JJ" for _, t in tagged[:2]))
    )
    feats["is_command_struct"] = float(
        any(t in VERB_TAGS for _, t in tagged[:1])
        and not any(t in SUBJECT_TAGS for _, t in tagged[:2])
    )
    feats["is_request_struct"] = float(
        first_word in {"would", "could", "can"}
        and any(t == "PRP" for _, t in tagged[1:2])
    )

    return feats


class LinguisticFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that converts text to a linguistic feature matrix.

    Extracts ~40 features per sentence (POS counts, bigram transitions, intent
    signals, lexical diversity) and encodes them as a sparse matrix via
    DictVectorizer.

    Can be composed in a ``FeatureUnion`` alongside TF-IDF vectorizers.

    Args:
        sparse: Return a sparse matrix (default True, compatible with
            ``scipy.sparse.hstack``).
    """

    def __init__(self, sparse: bool = True) -> None:
        self.sparse = sparse
        self._vec: DictVectorizer | None = None

    def fit(self, X: List[str], y=None) -> "LinguisticFeaturesTransformer":
        dicts = [sentence_to_features(s) for s in X]
        self._vec = DictVectorizer(sparse=self.sparse)
        self._vec.fit(dicts)
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        if self._vec is None:
            raise RuntimeError("Call fit() first.")
        dicts = [sentence_to_features(s) for s in X]
        return self._vec.transform(dicts)
