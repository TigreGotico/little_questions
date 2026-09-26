"""Shared constants for little_questions.

Single source of truth for sentence type labels, EAT question-type labels, and
supported language codes.
"""

SENTENCE_TYPES: list[str] = ["command", "exclamation", "question", "request", "statement"]

# EAT taxonomy — 7 main categories (incl. BOOL)
EAT_LABELS_7: list[str] = ["ABBR", "BOOL", "DESC", "ENTY", "HUM", "LOC", "NUM"]

# EAT taxonomy — 53 fine-grained labels (incl. BOOL:yesno)
EAT_LABELS_53: list[str] = [
    "ABBR:abb", "ABBR:exp",
    "BOOL:yesno",
    "DESC:def", "DESC:desc", "DESC:manner", "DESC:reason",
    "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:cremat",
    "ENTY:currency", "ENTY:dismed", "ENTY:event", "ENTY:food",
    "ENTY:instru", "ENTY:lang", "ENTY:letter", "ENTY:other",
    "ENTY:plant", "ENTY:product", "ENTY:religion", "ENTY:sport",
    "ENTY:substance", "ENTY:symbol", "ENTY:techmeth", "ENTY:termeq",
    "ENTY:veh", "ENTY:word",
    "HUM:desc", "HUM:gr", "HUM:ind", "HUM:title",
    "LOC:city", "LOC:country", "LOC:landmass", "LOC:mount",
    "LOC:other", "LOC:state", "LOC:water",
    "NUM:code", "NUM:count", "NUM:date", "NUM:dist",
    "NUM:money", "NUM:ord", "NUM:other", "NUM:perc",
    "NUM:period", "NUM:speed", "NUM:temp", "NUM:volsize", "NUM:weight",
]

MAIN_LABEL_NAMES: dict[str, str] = {
    "ABBR": "Abbreviation",
    "BOOL": "Boolean",
    "DESC": "Description",
    "ENTY": "Entity",
    "HUM": "Human",
    "LOC": "Location",
    "NUM": "Numeric",
}

SEC_LABEL_NAMES: dict[str, str] = {
    "yesno": "yes/no question",
    "abb": "abbreviation",
    "exp": "expression abbreviated",
    "def": "definition",
    "desc": "description",
    "manner": "manner",
    "reason": "reason",
    "animal": "animal",
    "body": "organs of body",
    "color": "color",
    "cremat": "inventions, books and other creative pieces",
    "currency": "currency",
    "dismed": "diseases and medicine",
    "event": "event",
    "food": "food",
    "instru": "instrument",
    "lang": "language",
    "letter": "letter",
    "other": "other",
    "plant": "plant",
    "product": "product",
    "religion": "religion",
    "sport": "sport",
    "substance": "substance",
    "symbol": "symbol",
    "techmeth": "technique or method",
    "termeq": "equivalent terms",
    "veh": "vehicles",
    "word": "word",
    "gr": "group or organization of persons",
    "ind": "individual",
    "title": "title",
    "city": "city",
    "country": "country",
    "landmass": "landmass",
    "mount": "mountain",
    "state": "state",
    "water": "body of water",
    "code": "code",
    "count": "count",
    "date": "date",
    "dist": "distance",
    "money": "money",
    "ord": "order",
    "perc": "percentage",
    "period": "period of time",
    "speed": "speed",
    "temp": "temperature",
    "volsize": "volume",
    "weight": "weight",
}
