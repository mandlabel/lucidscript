import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Set the competition name to download notebooks for
competition_name = "titanic"  # Example competition

# Create a directory to store notebooks
notebooks_dir = "notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

# 1. List Python notebooks for the competition
print("Searching and listing notebooks...")
notebooks = api.kernels_list(competition=competition_name, language='python', output_type='all')

print(f"Downloading notebooks for the '{competition_name}' competition...")
for i, notebook in enumerate(notebooks[:5]):  # Download the first 5 notebooks
    notebook_ref = notebook.ref
    print(f"Downloading notebook: {notebook_ref}")
    try:
        api.kernels_pull(kernel=notebook_ref, path=notebooks_dir)  # Download notebook as .ipynb
        print(f"Notebook downloaded: {notebook_ref}")
    except Exception as e:
        print(f"Error downloading {notebook_ref}: {e}")

print(f"Notebooks downloaded to the '{notebooks_dir}' directory.")
