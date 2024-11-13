import os
import re
import nbformat
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import jaccard_score
import numpy as np

# Paths to the folders and input script
notebooks_dir = './notebooks'
input_script_path = './input/sample_script.py'

# Defining patterns to detect common data preparation steps
patterns = {
    'fillna': r'\.fillna\(',
    'dropna': r'\.dropna\(',
    'scaling': r'StandardScaler|MinMaxScaler',
    'outlier_removal': r'outlier|drop.*outlier'
}

# Step 1: Process each notebook in the folder and extract common steps
corpus_steps = Counter()

print("Analyzing notebooks to identify common data preparation steps...")
for filename in os.listdir(notebooks_dir):
    if filename.endswith('.ipynb'):
        filepath = os.path.join(notebooks_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Go through each code cell in the notebook
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                code = cell.source
                # Check for each pattern in the code and count matches
                for step, pattern in patterns.items():
                    if re.search(pattern, code):
                        corpus_steps[step] += 1

# Displaying the most common steps in the corpus
print("Common data preparation steps in the corpus:")
for step, count in corpus_steps.items():
    print(f"{step}: {count} occurrences")

# Step 2: Analyze the input script for common data preparation steps
input_steps = {step: 0 for step in patterns}

print("\nAnalyzing the input script for data preparation steps...")
with open(input_script_path, 'r', encoding='utf-8') as f:
    code = f.read()
    for step, pattern in patterns.items():
        if re.search(pattern, code):
            input_steps[step] = 1  # Mark as present in the input script

# Step 3: Compute metrics
# Convert to frequency lists for entropy calculation
corpus_freq = [corpus_steps.get(step, 0) for step in patterns]
input_freq = [input_steps.get(step, 0) for step in patterns]

# Calculate relative entropy
divergence = entropy(input_freq, corpus_freq)
print(f"\nRelative Entropy (KL Divergence): {divergence:.2f}")

# Convert to presence arrays for Jaccard similarity
corpus_presence = np.array([1 if corpus_steps.get(step, 0) > 0 else 0 for step in patterns])
input_presence = np.array([1 if input_steps.get(step, 0) > 0 else 0 for step in patterns])

# Calculate Jaccard similarity
similarity = jaccard_score(corpus_presence, input_presence, average='binary')
print(f"Jaccard Similarity: {similarity:.2f}")

# Step 4: Compare input steps with the corpus steps and suggest standardizations
print("\nComparison between input script and corpus:")
for step in corpus_steps:
    in_corpus = corpus_steps[step]
    in_input_script = input_steps[step]
    print(f"{step} - Corpus occurrences: {in_corpus}, Present in input script: {bool(in_input_script)}")

# Step 5: Suggest standardization based on missing steps and threshold
suggestions = []
for step, present in input_steps.items():
    if present == 0 and corpus_steps[step] >= 3:  # Threshold: 3 occurrences in corpus
        suggestions.append(f"Consider adding '{step}' step, as it's commonly used in the corpus.")

print("\nStandardization Suggestions:")
for suggestion in suggestions:
    print(suggestion)

# Step 6: Final assessment based on metrics
if divergence > 0.5:
    print("\nNote: The input script significantly deviates from the corpus based on relative entropy.")
if similarity < 0.8:
    print("Note: The input script has a low similarity to the corpus based on the Jaccard index.")