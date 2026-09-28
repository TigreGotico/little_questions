import json
import requests
import os
import hashlib
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

#############
# LLM prompts for data augmentation
###################

SYSTEM_PROMPT = """
You are an expert NLP Data Scientist specialized in the TREC Question Taxonomy. 
Your goal is to perform data augmentation by generating high-quality, diverse questions.

### FULL TREC TAXONOMY
- **ABBR (Abbreviation):** abb (abbreviation), exp (expression abbreviated)
- **DESC (Description):** def (definition), desc (description), manner (manner of action), reason (reasons)
- **ENTY (Entity):** animal, body (organs), color, cremat (creations), currency, dismed (diseases/med), event, food, instru (instruments), lang (languages), letter (alphabet), other, plant, product, religion, sport, substance, symbol, techmeth (techniques), termeq (equivalent terms), veh (vehicles), word (specific words)
- **HUM (Human):** desc (person description), gr (group/org), ind (individual), title (person title)
- **LOC (Location):** city, country, landmass, mount (mountains), other, state, water (lakes/oceans)
- **NUM (Numeric):** code (postcodes/phone), count (number of), date, dist (distance/height), money, ord (ordinal numbers), other, perc (percentage), period (duration), speed, temp (temperature), volsize (volume/size), weight

### GENERATION GUIDELINES
1. **Target the Answer**: The question must logically lead to an answer of type {label}.
2. **Structural Diversity**: Mix question starters. Do not start every sentence with the same word (e.g., if the seeds use "What", try starting with "Which", "Name a", or "Can you tell me").
3. **Complexity Variance**: Generate a mix of short, direct questions and longer, more contextual questions.
4. **No Placeholders**: Do not use "Question X" or "[Insert Name]". Use real-world entities, places, and facts.
5. **Natural Language**: Ensure the questions sound like something a human would actually ask a search engine or assistant.

### INSTRUCTIONS
1. You will receive a 'Label' and 5 'Seed Questions'.
2. Generate exactly 5 NEW, unique questions for that SAME label.
3. Output your response strictly as a JSON list of strings.

### OUTPUT FORMAT
["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]

### EXAMPLE
Label: NUM:dist
Seeds:
- How tall is the Eiffel Tower?
- What is the width of a football field?
- How far is the moon from Earth?
- What is the depth of the Grand Canyon?
- How long is the Nile river?

Output:
[
  "What is the average height of a giraffe?",
  "How many miles is it from London to Paris?",
  "What is the wingspan of a Boeing 747?",
  "How deep is the deepest part of the Atlantic Ocean?",
  "What is the total length of the Great Wall of China?"
]
"""


def local_chat(messages: list[dict], max_tokens: int = 512) -> str:
    """Sends a request to the local LLM server."""
    r = requests.post(
        "http://192.168.1.200:8000/v1/chat/completions",
        json={"model": "local", "messages": messages, "max_tokens": max_tokens},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def generate_batch(label: str, seeds: list[str]) -> list[str]:
    """Sends a batch of seeds to the LLM to generate new questions."""
    seed_text = "\n".join([f"- {s}" for s in seeds])
    user_prompt = f"Label: {label}\nSeeds:\n{seed_text}\n\nOutput:"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    llm_output = local_chat(messages)
    try:
        # Extract JSON list from output
        start_idx = llm_output.find("[")
        end_idx = llm_output.rfind("]") + 1
        new_questions = json.loads(llm_output[start_idx:end_idx])
        return new_questions if isinstance(new_questions, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


# Configuration
BASE = "/home/miro/AgentWorkspaces/LILACS/little_questions/datasets"
INPUT_FILE = f"{BASE}/question_types_EN_validated.csv"
OUTPUT_FILE = f"{BASE}/question_types_EN_augmented.csv"
RESUME_STATE = f"{BASE}/augment_progress.json"
CONCURRENT_REQUESTS = 2

# 1. Load Data & Group by Label
data_by_label = defaultdict(list)
with open(INPUT_FILE, "r") as f:
    lines = f.read().splitlines()[1:]
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 3:
            label, question, lang = parts[0], parts[1], parts[2]
            data_by_label[label].append(question)

# 2. Resume Logic: Load progress
processed_hashes = set()
if os.path.exists(RESUME_STATE):
    with open(RESUME_STATE, "r") as f:
        processed_hashes = set(json.load(f))

# 3. Prepare Work Batches
work_batches = []
for label, questions in data_by_label.items():
    # Chunk existing questions into groups of 5
    for i in range(0, len(questions), 5):
        seeds = questions[i:i + 5]
        if len(seeds) < 3: continue  # Need at least a few seeds for quality

        # Create a unique hash for this seed batch to track progress
        batch_id = hashlib.md5(f"{label}:{''.join(seeds)}".encode()).hexdigest()
        if batch_id not in processed_hashes:
            work_batches.append((label, seeds, batch_id))

# 4. Open Output File
f_out = open(OUTPUT_FILE, "a", buffering=1)
if os.path.getsize(OUTPUT_FILE) == 0:
    f_out.write("label\tquestion\tlang\n")


def process_batch(batch):
    label, seeds, batch_id = batch
    try:
        new_qs = generate_batch(label, seeds)
        return {"id": batch_id, "label": label, "questions": new_qs}
    except Exception:
        return None


# 5. Parallel Execution
print(f"Starting augmentation. {len(work_batches)} batches to process.")
with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
    futures = {executor.submit(process_batch, b): b for b in work_batches}

    for future in tqdm(as_completed(futures), total=len(work_batches)):
        result = future.result()
        if result and result["questions"]:
            # Write new questions to CSV
            for q in result["questions"]:
                f_out.write(f"{result['label']}\t{q}\ten\n")

            # Update resume state
            processed_hashes.add(result["id"])
            with open(RESUME_STATE, "w") as f_state:
                json.dump(list(processed_hashes), f_state)

f_out.close()
print(f"Augmentation complete. Results saved to {OUTPUT_FILE}")