from train.lang.pt.tokenize import word_tokenize_pt

try:
    from brill_postagger import BrillPostagger

    HAS_BRILL = True
except ImportError:
    HAS_BRILL = False


def load_pt_tagger():
    if not HAS_BRILL:
        raise ImportError(
            "brill_postagger is required for Portuguese POS tagging. "
            "Install with: pip install brill_postagger"
        )
    return BrillPostagger.from_pretrained("pt")


def pos_tag_pt(tokens, tagger=None):
    tagger = tagger or load_pt_tagger()
    if isinstance(tokens, str):
        tokens = word_tokenize_pt(tokens)
    postagged = tagger.tag(tokens)

    DETS = ["a", "á", "o", "ós", "aos", "ao"]
    for idx, (w, t) in enumerate(postagged):
        if w.lower() in DETS and t == "NOUN":
            postagged[idx] = (w, "DET")

    return postagged
