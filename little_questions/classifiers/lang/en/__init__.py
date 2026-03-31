"""English sentence type classifier using trained model."""

from little_questions.sentence_type import SentenceTypeClassifier


def get_scorer():
    """Get the English sentence type scorer."""
    return SentenceTypeClassifier.get_instance()
