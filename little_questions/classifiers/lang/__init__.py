"""Language-specific sentence scorers.

get_scorer() is the canonical entry point; it is defined in
little_questions.classifiers and re-exported here for convenience.
"""

from little_questions.classifiers import get_scorer

__all__ = ["get_scorer"]
