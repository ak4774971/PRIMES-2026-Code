# MIT PRIMES 2026
MIT PRIMES 2026 code for experimentation.

The main training and evaluation script is `word2vec.py`, while `word2vec_curvature.py` contains the source code of some of the experiments we ran for Section 3.

## Python packages
Install the base dependencies with:
```bash
pip install -r requirements.txt
```

### GPU acceleration (CuPy)
`Neural.py` / `word2vect_experiments.py` use CuPy when available (otherwise they fall back to NumPy on CPU). Install a CuPy build that matches your CUDA version, for example:
```bash
nvidia-smi   # check the CUDA Version reported in the header
pip install cupy-cuda12x          # CUDA 12.x
# or
pip install "cupy-cuda13x[ctk]"   # CUDA 13.x (includes CUDA Toolkit wheels)
```
See the [CuPy install guide](https://docs.cupy.dev/en/stable/install.html) for other CUDA versions.

## Required Files and Datasets
To run `word2vec.py` successfully, the following files need to be downloaded in the same directory as the `word2vec.py` script:

* `AllCombined.txt`
    * The raw text corpus to train word embeddings, using the Simple English Wikipedia
    * You can download it through this [link](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish)
* `SimLex-999.txt`
    * The SimLex-999 dataset which measures how well the embeddings capture semantic similarity
    * You can download it through this [link](https://fh295.github.io/SimLex-999.zip). This is a zip file that contains the dataset itself and a README file that contains a detailed description of the dataset.
* `wordsim353crowd.csv`
    * The WordSim-353 dataset which measures how well the embeddings understand general word relatedness
    * You can download it through this [link](https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd). Alternatively, you can also download [link](https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip), which contains slightly different similarity scores.
* `questions-words.txt`
    * The Google Analogy Test Set which measures how well the embeddings comprehend the general relationships between words
    * You can find the dataset through this [link](http://download.tensorflow.org/data/questions-words.txt).

### Quick setup with `download_files.py`
1. Install the required package:
   ```bash
   pip install kaggle
   ```
2. Create a Kaggle API token from [Kaggle Account Settings](https://www.kaggle.com/settings) (`Create New API Token`), then export it in your shell:
   ```bash
   export KAGGLE_API_TOKEN="<your_kaggle_token>"
   ```
3. From the project root, download all datasets:
   ```bash
   python download_files.py
   ```
