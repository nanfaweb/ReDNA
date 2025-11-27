# ReDNA
Applying natural language processing techniques to sets of DNA strands, aiming to classify strings by training models.

## Install dependencies

This project uses a virtual environment located at `.venv` by default. To install the Python dependencies run (PowerShell):

```powershell
# activate the venv
. .\.venv\Scripts\Activate.ps1

# install project dependencies
pip install -r requirements.txt
```

If you're running the notebook directly, there's a small helper cell at the top of `dataset_creation.ipynb` that will auto-install `pandas` and `biopython` in the running kernel.
