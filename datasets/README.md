---
license: mit
task_categories:
- text-classification
language:
- en
tags:
- trec
- question-classification
- eat
- semantic
- intent-detection
- synthetic-data
size_categories:
- 1K<n<10K
---

# EAT: Expected Answer Type Dataset

A high-quality dataset for Question Classification based on the **TREC Question Taxonomy**, enhanced with modern categories and strict Expected Answer Type (EAT) validation.

## Dataset Summary
The **EAT (Expected Answer Type)** dataset is designed to train and evaluate NLP models in the task of classifying questions not by their surface keywords, but by the semantic category of their expected answer. 

Unlike original TREC datasets, this version includes:
1.  **BOOL:yesno** category for polar questions.
2.  **Verified Answer Column** for every sample to ensure grounding and label accuracy.
3.  **Structural Diversity** (Direct, Modal, Imperative, and Contextual phrasings).
4.  **Length Stratification** (Balanced distribution of Short, Medium, Long, and Very Long samples).

## Dataset Generation
This dataset is **synthetically generated using Large Language Models (LLM)** under the strict supervision of a human data engineer. Every entry was produced by providing the LLM with the TREC taxonomy definitions and the rigorous generation guidelines listed below. An iterative "Generate-Validate-Clean" loop was used to ensure the expected answer strictly matches the assigned label.

## Dataset Structure
The dataset is provided in **TSV (Tab-Separated Values)** format with the following columns:

| Column | Description |
| :--- | :--- |
| `label` | The hierarchical TREC label (e.g., `NUM:dist`, `HUM:ind`). |
| `question` | The natural language question or command. |
| `answer` | A factual, representative answer used to verify the label. |
| `lang` | The language code (currently `en`). |

## Classification Taxonomy

### Main Labels
*   **ABBR** (Abbreviation)
*   **BOOL** (Boolean/Yes-No)
*   **DESC** (Description)
*   **ENTY** (Entity)
*   **HUM** (Human)
*   **LOC** (Location)
*   **NUM** (Numeric)

### Hierarchical Labels (Secondary)
*   **ABBR:** `abb`, `exp`
*   **BOOL:** `yesno`
*   **DESC:** `def`, `desc`, `manner`, `reason`
*   **ENTY:** `animal`, `body`, `color`, `cremat`, `currency`, `dismed`, `event`, `food`, `instru`, `lang`, `letter`, `other`, `plant`, `product`, `religion`, `sport`, `substance`, `symbol`, `techmeth`, `termeq`, `veh`, `word`
*   **HUM:** `desc`, `gr`, `ind`, `title`
*   **LOC:** `city`, `country`, `landmass`, `mount`, `other`, `state`, `water`
*   **NUM:** `code`, `count`, `date`, `dist`, `money`, `ord`, `other`, `perc`, `period`, `speed`, `temp`, `volsize`, `weight`

## LLM Generation Guidelines
Every sample was generated according to these six fundamental mandates provided to the model:

1.  **The EAT Anchor:** Categorization is determined solely by the answer type. (e.g., "How high is Everest?" is `NUM:dist` because the answer is a measurement).
2.  **Structural Diversity:** Phrasing must vary between direct ("What is...?"), imperative ("Describe the process of..."), and modal ("Can you tell me if...?").
3.  **Length Stratification:** Every label must follow a 10/20/10/5 ratio for Short (concise), Medium (standard), Long (detailed), and Very Long (complex/analytical) queries.
4.  **Entity Granularity:** Use specific real-world entities (e.g., "Mariana Trench") rather than generic subjects.
5.  **Grammatical Rigor:** Strict adherence to proper capitalization, punctuation, and correct technical terminology.
6.  **Boundary Testing:** Inclusion of "hard negatives" to help models distinguish similar labels (e.g., distinguishing `LOC:mount` from `NUM:dist`).

---
**Maintained by:** [TigreGotico](https://huggingface.co/TigreGotico)  
**Status:** In Progress (Targeting 500 samples per label).
