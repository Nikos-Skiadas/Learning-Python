# Data Mining Assignment

Multimodal music genre classification pipeline. Combines audio features (MFCC statistics), lyrics (sentence transformer embeddings), genre labels (binary encoding), and user tags to build a rich representation of songs for downstream analysis.

## Data

Place the following raw files in `data/`:

| File | Description |
| ---- | ----------- |
| `id_mfcc_stats.tsv.bz2` | MFCC means (13) and covariance matrix (91 entries) per song |
| `processed_lyrics.tar.gz` | Preprocessed/stemmed lyrics, one text file per song |
| `id_genres.csv` | Comma-separated genre labels per song (multilabel) |
| `id_tags.csv` | Comma-separated user tags per song |
| `id_information.csv` | Song metadata: artist, song name, album |

All TSV/CSV files are tab-separated with the song ID as the first column.

## Pipeline

### 1. Dataset Creation (`src/main.py`)

Loads raw data, computes the intersection of all sources (keeping only songs present in every file with no missing values), generates embeddings, and caches everything to disk.

```bash
python -m project.src.main -k 5 --epochs 256 -- project/data
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `data` | required | Path to the data directory |
| `-k` | `5` | Number of top genres to keep (by frequency) |
| `--epochs` | `1` | Training epochs for the audio autoencoder |
| `--model` | `all-MiniLM-L6-v2` | Sentence transformer model for lyric embeddings |
| `--force` | off | Ignore cached files and regenerate everything |

**Outputs** (cached in `data/`):

- `dataset.{k}.csv` — the filtered dataset with all raw columns
- `dataset.{k}.genres.parquet` — binary one-hot genre encodings
- `dataset.{k}.tags.parquet` — binary one-hot tag encodings
- `dataset.{k}.audio.parquet` — autoencoder-compressed audio embeddings
- `dataset.{k}.lyrics.parquet` — 384-dim sentence transformer embeddings

Subsequent runs skip generation if all files already exist (unless `--force` is passed).

### 2. Exploratory Data Analysis (`src/eda.py`)

Loads the cached dataset and embeddings, then runs all visualization tasks.

```bash
python -m project.src.eda -k 5 --output project/figures -- project/data
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `data` | required | Path to the data directory |
| `-k` | `5` | Number of top genres |
| `--output` | `None` | Directory to save figures as PNG (shows interactively if omitted) |

**Visualizations produced:**

- **Word Clouds** — tag clouds for the two most different genres (by tag cosine distance)
- **Top Tags** — bar chart of the 10 most frequent tags
- **t-SNE Scatter** — side-by-side 2D projections of audio vs lyrics embeddings, colored by genre
- **Genre Count Histogram** — how many genres each song belongs to
- **Genre Distribution** — song counts for the top genres
- **Lyrics Length** — character count, word count, and meaningful word count (stopwords removed)
- **Sentiment by Genre** — DistilBERT sentiment score distributions per genre (violin plot)
- **Similarity Search** — top-5 most similar songs by cosine similarity on lyrics and audio embeddings

All plot functions are independently callable and accept an optional `ax` parameter for composition into custom figures.

### 3. Multi-label Classification (`src/classification.py`)

Runs the Part B experiments while preserving the fact that each song can belong to multiple genres. The target matrix keeps the top genre labels as separate binary targets instead of collapsing each song to a single class.

```bash
python -m project.src.classification -k 5 -- project/data
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `data` | required | Path to the cached data directory |
| `-k` | `5` | Dataset cache to load |
| `--labels` | same as `-k` | Number of top labels to predict |
| `--folds` | `10` | Cross-validation folds |
| `--classifier` | `logistic` | `logistic` or `random_forest` |
| `--regularization-c` | `None` | Optional override for all logistic C values |
| `--text-c` | `1.0` | Logistic C for text-only features |
| `--audio-c` | `10.0` | Logistic C for audio-only features |
| `--early-c` | `1.0` | Logistic C for concatenated early-fusion features |
| `--bilinear-c` | `0.1` | Logistic C for bilinear pooling features |
| `--threshold` | `0.5` | Probability threshold for assigning labels |
| `--max-samples` | `None` | Optional smaller sample for quick checks |
| `--skip-bilinear` | off | Skip the bilinear pooling ablation |
| `--skip-clustering` | off | Skip K-Means evaluation |
| `--output` | `None` | Optional directory for metrics CSV files |

**Multi-label adaptation of Part B:**

- **Text-only / Audio-only**: train one-vs-rest multi-label classifiers on lyric and audio embeddings.
- **Early Fusion**: concatenate text and audio embeddings before training.
- **Bilinear Pooling / Outer-product Fusion**: use only all pairwise text-audio products as interaction features. With 384 lyric dimensions and 11 audio dimensions, this produces `384 * 11 = 4224` cross-modal features per song.
- **Late Fusion**: average the per-label probabilities from the text-only and audio-only models, then threshold them.
- **Metrics**: report subset accuracy, Hamming loss, macro/micro precision, recall, F1, sample-wise F1, and sample-wise Jaccard.
- **Confusion Matrices**: use one binary 2x2 confusion matrix per genre label.
- **Clustering**: run K-Means on fused embeddings, use Silhouette Score, and compare clusters to multi-label ground truth through label-set ARI and average per-label ARI.

Cross-validation is used as the evaluation protocol: every fixed model configuration is tested through the same folds. It is not used here as an automatic hyperparameter search. The logistic `C` values are chosen upfront from the dimensionality of each representation.

**Bilinear pooling ablation note:**

Bilinear pooling, also called outer-product fusion, models cross-modal interactions by forming `text_i * audio_j` for every lyric/audio feature pair. In this project we use the cross-products alone for the ablation, rather than appending them to the concatenated vector. This tests whether interaction information is useful by itself, while the standard early-fusion baseline remains simple concatenation.

Because the feature dimensions differ substantially across experiments, regularization matters. The defaults use weaker regularization for the compact audio vector (`--audio-c 10.0`), standard regularization for text and concatenated early fusion (`--text-c 1.0`, `--early-c 1.0`), and stronger regularization for the 4224-dimensional bilinear representation (`--bilinear-c 0.1`). Use `--regularization-c` only when intentionally forcing one shared value across all logistic models.

**Late fusion ablation note:**

The implemented solution keeps late fusion as soft consensus: for each label, average the text and audio probabilities and threshold the result. Because multi-label predictions are set-valued binary decisions, two useful ablation baselines are probabilistic OR and probabilistic AND:

- **Probabilistic OR**: `1 - (1 - p_text) * (1 - p_audio)`, a permissive fuzzy-union rule that usually increases recall.
- **Probabilistic AND**: `p_text * p_audio`, a conservative fuzzy-intersection rule that usually increases precision.

Soft consensus sits between these two rules for probabilities in `[0, 1]`, so it is a balanced default rather than a strict union or intersection.

For hard binary modality decisions, averaging has a borderline behavior: with a `>= 0.5` threshold, a disagreement `(1, 0)` or `(0, 1)` is accepted and the rule behaves like set union; with a strict `> 0.5` threshold, disagreement is rejected and the rule behaves like set intersection. With probabilistic outputs, averaging is better interpreted as soft consensus, and the threshold controls how permissive or conservative the resulting label set is.

## Module Overview

```text
src/
  data.py      MusicSeries / MusicDataFrame — pandas subclasses for loading,
               filtering, and intersecting the raw data files
  encoding.py  AudioAutoencoder (PyTorch) — unsupervised dimensionality reduction
               for MFCC features; sentence transformer wrapper for lyrics;
               binary encoding for multilabel genres/tags
  main.py      Dataset orchestration — loads raw data, generates and caches
               embeddings, CLI entry point
  eda.py       EDA visualizations — word clouds, distribution plots, t-SNE,
               sentiment analysis, similarity search
  classification.py
               Multi-label classification, fusion, evaluation, confusion
               matrices, and K-Means comparison for Part B
```

## Dependencies

- `pandas`, `numpy` — data handling
- `torch` — audio autoencoder
- `sentence-transformers` — lyrics embeddings (`all-MiniLM-L6-v2`)
- `transformers` — sentiment analysis pipeline (DistilBERT)
- `scikit-learn` — t-SNE, cosine similarity, classifiers, metrics, K-Means
- `matplotlib`, `seaborn` — plotting
- `wordcloud` — word cloud generation
- `nltk` — stopword lists (download with `python -c "import nltk; nltk.download('stopwords')"`)
- `rich` — progress bars for autoencoder training
