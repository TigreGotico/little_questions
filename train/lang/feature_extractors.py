"""Language-specific categorical feature extractors for sentence-type classification.

Extracts dict-based linguistic features (intent signals, lexical metrics, punctuation)
from raw text without NLTK or external dependencies. Features are sklearn-compatible
and designed to embed in ONNX models.

Usage:
    extractor = LanguageFeatureExtractor_EN()
    features = extractor.extract("What is the capital of France?")
    # → {'starts_wh': 1.0, 'ends_question': 1.0, 'sentence_length': 6, ...}

Each language subclass (EN, ES, FR, etc.) defines:
- Language-specific keyword dicts (WH_STARTERS, POLITE_WORDS, etc.)
- Custom extraction rules tailored to that language's syntax
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class LanguageFeatureExtractor(ABC):
    """Abstract base for language-specific feature extractors.

    Subclasses must implement extract() and define language-specific keyword dicts.
    """

    lang: str = "en"

    @abstractmethod
    def extract(self, text: str) -> Dict[str, float]:
        """Extract categorical features from raw text.

        Args:
            text: Raw input sentence

        Returns:
            Dict mapping feature name → numeric value (bool as 0/1 float)
        """
        pass


class LanguageFeatureExtractor_EN(LanguageFeatureExtractor):
    """English sentence-type feature extractor."""

    lang = "en"

    # ---- Language-specific keyword dicts (must be immutable for ONNX) ----
    WH_STARTERS = frozenset({
        "what", "where", "when", "why", "how", "who", "which", "whom", "whose"
    })

    POLITE_STARTERS = frozenset({
        "would", "could", "can", "may", "might", "please"
    })

    COMMAND_VERBS = frozenset({
        "go", "come", "stop", "start", "do", "make", "let", "put", "get", "give",
        "take", "run", "walk", "talk", "listen", "watch", "look", "see", "try",
        "help", "find", "open", "close", "turn", "move", "sit", "stand", "wait",
        "remember", "forget", "ensure", "check"
    })

    EXCLAMATION_MARKERS = frozenset({
        "what", "how"  # "what a" or "how beautiful" patterns
    })

    NEGATION_WORDS = frozenset({
        "no", "not", "never", "neither", "nobody", "nothing", "nowhere"
    })

    POLITE_WORDS = frozenset({
        "please", "thank", "thanks", "kindly", "would", "could", "might", "sorry"
    })

    REQUEST_OPENERS = frozenset({
        "would", "could", "can", "may", "might", "please"
    })

    def _tokenize(self, text: str) -> list[str]:
        """Simple regex-based tokenization (no NLTK)."""
        # Simple word splitting on whitespace and punctuation
        words = re.findall(r"\b\w+\b", text.lower())
        return words

    def extract(self, text: str) -> Dict[str, float]:
        """Extract features from English text."""
        text_lower = text.lower().strip()
        tokens = self._tokenize(text_lower)
        feats: Dict[str, float] = {}

        if not tokens:
            return feats

        first_word = tokens[0]

        # ---- Intent signals ----
        feats["starts_wh"] = float(first_word in self.WH_STARTERS)
        feats["starts_polite"] = float(first_word in self.POLITE_STARTERS)
        feats["starts_command"] = float(first_word in self.COMMAND_VERBS)
        feats["starts_exclamation"] = float(first_word in self.EXCLAMATION_MARKERS)

        # ---- Punctuation signals ----
        feats["ends_question"] = float(text.rstrip().endswith("?"))
        feats["ends_exclamation"] = float(text.rstrip().endswith("!"))
        feats["ends_period"] = float(text.rstrip().endswith("."))
        feats["has_ellipsis"] = float("..." in text)
        feats["multiple_punctuation"] = float(
            sum(1 for c in text if c in "?!") > 1
        )

        # ---- Negation ----
        feats["has_negation"] = float(any(w in tokens for w in self.NEGATION_WORDS))

        # ---- Lexical features ----
        feats["sentence_length"] = float(len(tokens))
        feats["unique_token_count"] = float(len(set(tokens)))
        feats["lexical_diversity"] = float(len(set(tokens)) / len(tokens)) if tokens else 0.0
        feats["avg_token_length"] = float(
            sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
        )

        # ---- Structural features ----
        feats["has_polite_words"] = float(
            any(w in self.POLITE_WORDS for w in tokens)
        )
        feats["is_short"] = float(len(tokens) <= 3)
        feats["is_long"] = float(len(tokens) >= 15)

        return feats


class LanguageFeatureExtractor_ES(LanguageFeatureExtractor):
    """Spanish sentence-type feature extractor."""

    lang = "es"

    WH_STARTERS = frozenset({
        "qué", "dónde", "cuándo", "por qué", "cómo", "quién", "cuál", "cuáles",
        "cuyo", "cuyos", "cuya", "cuyas"
    })

    POLITE_STARTERS = frozenset({
        "podría", "podría", "puede", "puedo", "pudiera", "podrías", "podrían",
        "por favor", "favor"
    })

    COMMAND_VERBS = frozenset({
        "ve", "ven", "vaya", "vayan", "vayas", "vamos", "ir",
        "corre", "corran", "correr", "corramos",
        "salta", "salten", "saltar",
        "haz", "hace", "hagan", "hagamos",
        "da", "den", "dar", "demos",
        "abre", "abran", "abrir", "abramos",
        "cierra", "cierren", "cerrar", "cerremos",
        "espera", "esperen", "esperar",
        "busca", "busquen", "buscar",
        "toma", "tomen", "tomar",
        "sigue", "sigan", "seguir",
        "ayuda", "ayuden", "ayudar",
        "recuerda", "recuerden", "recordar",
        "olvida", "olviden", "olvidar",
        "asegura", "aseguren", "asegurar"
    })

    EXCLAMATION_MARKERS = frozenset({
        "qué", "cómo"  # "¡Qué..." or "¡Cómo..." patterns
    })

    NEGATION_WORDS = frozenset({
        "no", "nunca", "jamás", "nada", "nadie", "ningún", "ninguno", "ninguna"
    })

    POLITE_WORDS = frozenset({
        "por favor", "favor", "gracias", "graciitas", "disculpa",
        "podría", "puede", "puedo", "pudiera"
    })

    REQUEST_OPENERS = frozenset({
        "podría", "puede", "puedo", "pudiera", "por favor"
    })

    def _tokenize(self, text: str) -> list[str]:
        """Simple regex-based tokenization."""
        words = re.findall(r"\b\w+\b", text.lower())
        return words

    def extract(self, text: str) -> Dict[str, float]:
        """Extract features from Spanish text."""
        text_lower = text.lower().strip()
        tokens = self._tokenize(text_lower)
        feats: Dict[str, float] = {}

        if not tokens:
            return feats

        first_word = tokens[0]

        # ---- Intent signals ----
        feats["starts_wh"] = float(first_word in self.WH_STARTERS)
        feats["starts_polite"] = float(first_word in self.POLITE_STARTERS)
        feats["starts_command"] = float(first_word in self.COMMAND_VERBS)
        feats["starts_exclamation"] = float(first_word in self.EXCLAMATION_MARKERS)

        # ---- Punctuation signals ----
        feats["ends_question"] = float(
            text.rstrip().endswith("?") or text.rstrip().endswith("?»")
        )
        feats["ends_exclamation"] = float(
            text.rstrip().endswith("!") or text.rstrip().endswith("!»")
        )
        feats["ends_period"] = float(text.rstrip().endswith("."))
        feats["has_ellipsis"] = float("..." in text)
        feats["multiple_punctuation"] = float(
            sum(1 for c in text if c in "?!") > 1
        )

        # ---- Negation ----
        feats["has_negation"] = float(any(w in tokens for w in self.NEGATION_WORDS))

        # ---- Lexical features ----
        feats["sentence_length"] = float(len(tokens))
        feats["unique_token_count"] = float(len(set(tokens)))
        feats["lexical_diversity"] = float(len(set(tokens)) / len(tokens)) if tokens else 0.0
        feats["avg_token_length"] = float(
            sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
        )

        # ---- Structural features ----
        feats["has_polite_words"] = float(
            any(w in self.POLITE_WORDS for w in tokens)
        )
        feats["is_short"] = float(len(tokens) <= 3)
        feats["is_long"] = float(len(tokens) >= 15)

        return feats


# Language factory mapping
FEATURE_EXTRACTORS = {
    "en": LanguageFeatureExtractor_EN,
    "es": LanguageFeatureExtractor_ES,
    "ca": LanguageFeatureExtractor_EN,  # Catalan: use EN template for now
    "fr": LanguageFeatureExtractor_EN,  # French: use EN template for now
    "de": LanguageFeatureExtractor_EN,  # German: use EN template for now
    "it": LanguageFeatureExtractor_EN,  # Italian: use EN template for now
    "nl": LanguageFeatureExtractor_EN,  # Dutch: use EN template for now
    "pt": LanguageFeatureExtractor_EN,  # Portuguese: use EN template for now
}


def get_feature_extractor(lang: str) -> LanguageFeatureExtractor:
    """Get feature extractor instance for language."""
    lang = lang.lower()
    extractor_class = FEATURE_EXTRACTORS.get(lang, LanguageFeatureExtractor_EN)
    return extractor_class()


class LanguageFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping language-specific feature extraction.

    Converts list of texts → feature matrix via DictVectorizer.
    Compatible with Pipeline and FeatureUnion for composition with TF-IDF.

    Args:
        lang: Language code (en, es, fr, etc.)
        sparse: Return sparse matrix (default True, compatible with TF-IDF + LinearSVC)
    """

    def __init__(self, lang: str = "en", sparse: bool = True) -> None:
        self.lang = lang.lower()
        self.sparse = sparse
        self._extractor: LanguageFeatureExtractor | None = None
        self._vectorizer: DictVectorizer | None = None

    def fit(self, X: list[str], y=None) -> LanguageFeatureTransformer:
        """Fit the feature extractor and DictVectorizer.

        Args:
            X: List of text samples
            y: Labels (unused, for sklearn compatibility)

        Returns:
            self
        """
        from sklearn.feature_extraction import DictVectorizer

        self._extractor = get_feature_extractor(self.lang)
        dicts = [self._extractor.extract(text) for text in X]
        self._vectorizer = DictVectorizer(sparse=self.sparse)
        self._vectorizer.fit(dicts)
        return self

    def transform(self, X: list[str]):
        """Transform texts to feature matrix.

        Args:
            X: List of text samples

        Returns:
            Feature matrix (sparse or dense, shape (n_samples, n_features))
        """
        if self._extractor is None or self._vectorizer is None:
            raise RuntimeError("Call fit() before transform().")

        dicts = [self._extractor.extract(text) for text in X]
        return self._vectorizer.transform(dicts)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for ONNX export."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before get_feature_names_out().")
        return self._vectorizer.get_feature_names_out(input_features)
