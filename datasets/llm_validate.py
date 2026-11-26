import json
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

#############
# LLM prompts to validate data
###################

SYSTEM_PROMPT = """
You are an expert NLP Classification Validator specialized in the TREC Question Taxonomy. 
Your goal is to verify if a question is correctly categorized based on its **Expected Answer Type (EAT)**.

### CLASSIFICATION TAXONOMY

MAIN LABELS:
- ABBR (Abbreviation)
- DESC (Description)
- ENTY (Entity)
- HUM (Human)
- LOC (Location)
- NUM (Numeric)

SECONDARY LABELS (Hierarchical):
- ABBR: abb, exp
- DESC: def, desc, manner, reason
- ENTY: animal, body, color, cremat, currency, dismed, event, food, instru, lang, letter, other, plant, product, religion, sport, substance, symbol, techmeth, termeq, veh, word
- HUM: desc, gr, ind, title
- LOC: city, country, landmass, mount, other, state, water
- NUM: code, count, date, dist, money, ord, other, perc, period, speed, temp, volsize, weight

### CORE CLASSIFICATION LOGIC
Classify based on what the **answer** would be:
- "How tall is Mount Everest?" -> NUM:dist
- "Where is Mount Everest?" -> LOC:other
- "Who climbed Everest?" -> HUM:ind

### RULES
1. **Validation**: Check if 'Proposed Label' matches expected answer type.
2. **Correction**: If incorrect, provide accurate labels from taxonomy above.
3. **Output**: Return ONLY a JSON object. No conversational filler.

### OUTPUT SCHEMA
{
  "is_valid": boolean,
  "reason": "explanation",
  "fixed_labels": {
    "main": "STRING_UPPER",
    "secondary": "string_lower"
  }
}
"""


def local_chat(messages: list[dict], max_tokens: int = 256) -> str:
    """Sends a request to the local LLM server."""
    r = requests.post(
        "http://192.168.1.200:8000/v1/chat/completions",
        json={"model": "local", "messages": messages, "max_tokens": max_tokens},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def validate_and_fix(question: str, label: str) -> dict:
    """Task function for the thread pool to execute."""
    prompt = f"label: {label}\nquestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    llm_output = local_chat(messages, max_tokens=512)
    return json.loads(llm_output)


# Configuration
BASE = "/home/miro/AgentWorkspaces/LILACS/little_questions/datasets"
DIRTY_DATA = f"{BASE}/question_types_EN.csv"
OUT_FULL = f"{BASE}/question_types_EN_full.csv"
OUT_FIXED = f"{BASE}/question_types_EN_fixed.csv"
OUT_VALIDATED = f"{BASE}/question_types_EN_validated.csv"
CONCURRENT_REQUESTS = 2

# Storage
PROCESSED_QUESTIONS = set()

# Resume Logic: Populate PROCESSED_QUESTIONS from existing output
if os.path.exists(OUT_FULL):
    with open(OUT_FULL, "r") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) > 1:
                PROCESSED_QUESTIONS.add(parts[1])  # Question text

# Open files in append mode to support parallel writing and resume
f_full = open(OUT_FULL, "a", buffering=1)
f_fixed = open(OUT_FIXED, "a", buffering=1)
f_valid = open(OUT_VALIDATED, "a", buffering=1)

# Write headers if files are new
if os.path.getsize(OUT_FULL) == 0:
    f_full.write("label\tquestion\tlang\tvalid\tfixed_label\treason\n")
if os.path.getsize(OUT_FIXED) == 0:
    f_fixed.write("label\tquestion\tlang\n")
if os.path.getsize(OUT_VALIDATED) == 0:
    f_valid.write("label\tquestion\tlang\n")


def process_row(row_data):
    """Worker function to process a single row."""
    label, question, lang = row_data
    if question in PROCESSED_QUESTIONS:
        return None

    try:
        res = validate_and_fix(question, label)
        fixed_l = f"{res['fixed_labels']['main']}:{res['fixed_labels']['secondary']}"

        return {
            "full": f"{label}\t{question}\t{lang}\t{res['is_valid']}\t{fixed_l}\t{res['reason']}",
            "fixed": f"{fixed_l}\t{question}\t{lang}",
            "valid": f"{label}\t{question}\t{lang}" if res['is_valid'] else None
        }
    except Exception:
        return None


# Load source data
with open(DIRTY_DATA) as f:
    rows = [l.split("\t", 2) for l in f.read().splitlines()[1:] if l.strip()]

print(f"Starting parallel processing with {CONCURRENT_REQUESTS} workers.")

with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
    # Map rows to the process_row function
    futures = {executor.submit(process_row, row): row for row in rows}

    for future in tqdm(as_completed(futures), total=len(rows)):
        result = future.result()
        if result:
            f_full.write(result["full"] + "\n")
            f_fixed.write(result["fixed"] + "\n")
            if result["valid"]:
                f_valid.write(result["valid"] + "\n")

# Cleanup
f_full.close()
f_fixed.close()
f_valid.close()