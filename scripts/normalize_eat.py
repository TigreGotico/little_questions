import os
import sys

def normalize_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    expected_columns = 4
    header = "label\tquestion\tanswer\tlang"
    unique_questions = set()
    valid_rows = []
    dropped_comments = 0
    dropped_malformed = 0
    dropped_duplicates = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            
            # 1. Skip empty lines or comments
            if not line or line.startswith('#'):
                if line.startswith('#'): dropped_comments += 1
                continue
            
            # 2. Skip header if it exists in the middle of file
            if line.startswith('label\tquestion'):
                continue

            parts = line.split('\t')
            
            # 3. Validate column count
            if len(parts) != expected_columns:
                dropped_malformed += 1
                continue
            
            # 4. Deduplicate based on question text (case-insensitive)
            label, question, answer, lang = parts
            q_norm = question.strip().lower()
            if q_norm in unique_questions:
                dropped_duplicates += 1
                continue
            
            unique_questions.add(q_norm)
            valid_rows.append([label.strip(), question.strip(), answer.strip(), lang.strip()])

        # 5. Sort by the 'question' column (index 1) for easier diffs
        valid_rows.sort(key=lambda x: x[1].lower())

        # Write cleaned data back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header + '\n')
            for row in valid_rows:
                f.write("\t".join(row) + '\n')

        print(f"Normalization Complete for {file_path}")
        print(f"- Total unique samples: {len(valid_rows)}")
        print(f"- Dropped comments: {dropped_comments}")
        print(f"- Dropped malformed rows: {dropped_malformed}")
        print(f"- Dropped duplicates: {dropped_duplicates}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_file = "/home/miro/AgentWorkspaces/ML/NLP/EAT/EAT.tsv"
    normalize_dataset(target_file)
