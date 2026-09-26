import re


def word_tokenize_pt(sentence):
    tokens_regex = re.compile(r"([., :;\n()\"#!?1234567890/&%+])",
                              flags=re.IGNORECASE)
    tokens = re.split(tokens_regex, sentence)
    postprocess = []
    postprocess_regex = re.compile(
        r"\b(\w+)-(me|te|se|nos|vos|o|os|a|as|lo|los|la|las|lhe|lhes|lha|lhas|lho|lhos|no|na|nas|mo|ma|mos|mas|to|ta|tos|tas)\b",
        flags=re.IGNORECASE)
    for token in tokens:
        for token2 in re.split(postprocess_regex, token):
            if token2.strip():
                postprocess.append(token2)

    return postprocess
