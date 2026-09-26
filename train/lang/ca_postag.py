from nltk import word_tokenize

try:
    from brill_postagger import BrillPostagger

    HAS_BRILL = True
except ImportError:
    HAS_BRILL = False


def load_ca_tagger():
    if not HAS_BRILL:
        raise ImportError(
            "brill_postagger is required for Catalan POS tagging. "
            "Install with: pip install brill_postagger"
        )
    return BrillPostagger.from_pretrained("ca")


def pos_tag_ca(tokens, tagger=None):
    tagger = tagger or load_ca_tagger()
    if isinstance(tokens, str):
        tokens = word_tokenize(tokens)
    postagged = tagger.tag(tokens)

    return postagged
