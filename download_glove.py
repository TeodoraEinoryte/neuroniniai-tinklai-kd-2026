"""Used once for S12: downloads GloVe 6B (Pennington et al., 2014) and keeps the 100-dimensional vectors in
data/glove/glove.6B.100d.txt (~330 MB, git-ignored). data.py reads that file.

Source: the Stanford NLP group's copy on Hugging Face (stanfordnlp/glove, glove.6B.zip, 862 MB; the same zip as
nlp.stanford.edu/data/glove.6B.zip). Only the 100-d file is kept.

Run: python download_glove.py
"""
import os
import zipfile

from huggingface_hub import hf_hub_download

import config

GLOVE_DIR = os.path.join(config.DATA_DIR, "glove")
GLOVE_FILE = "glove.6B.100d.txt"

if __name__ == "__main__":
    os.makedirs(GLOVE_DIR, exist_ok=True)
    zip_path = hf_hub_download("stanfordnlp/glove", "glove.6B.zip", local_dir=GLOVE_DIR)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(GLOVE_FILE, GLOVE_DIR)
    os.remove(zip_path)                               # 862 MB; the 50-, 200- and 300-d files are not needed
    with open(os.path.join(GLOVE_DIR, GLOVE_FILE), encoding="utf-8") as f:
        n = sum(1 for _ in f)
    print(f"Išsaugota: {os.path.join(GLOVE_DIR, GLOVE_FILE)} ({n} žodžių)")
