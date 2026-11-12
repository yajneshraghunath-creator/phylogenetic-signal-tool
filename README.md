# Protein Sequence Alignment-Based Phylogenetic Signal Tool

Streamlit app (Group 16): FASTA protein input → validation → pairwise alignment (Biopython
`PairwiseAligner`; `pairwise2` is deprecated) → distance matrix heatmap → NJ/UPGMA tree →
phylogenetic signal (Blomberg's K with permutation test) → interpretation.

## Run locally
- macOS/Linux: `./run.sh`
- Windows: double-click `run.bat`
- Manual: `pip install -r requirements.txt && streamlit run app.py`

Then open http://localhost:8501.

## Files
- `core.py` – bioinformatics logic (no UI)
- `app.py` – Streamlit interface
- `test_core.py` – tests (`python -m pytest` or `python test_core.py`)

## Notes
- "Load simulated demo" uses synthetic sequences/traits, not real proteins.
- Trait file format: `name,value` per line, names must match FASTA headers.
- Distances: p-distance (1 − identity) or Poisson-corrected. Signal tests need ≥3 taxa; power is low with few taxa.
