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
```

## Dependencies

- `pandas`, `numpy` — data handling
- `torch` — audio autoencoder
- `sentence-transformers` — lyrics embeddings (`all-MiniLM-L6-v2`)
- `transformers` — sentiment analysis pipeline (DistilBERT)
- `scikit-learn` — t-SNE, cosine similarity
- `matplotlib`, `seaborn` — plotting
- `wordcloud` — word cloud generation
- `nltk` — stopword lists (download with `python -c "import nltk; nltk.download('stopwords')"`)
- `rich` — progress bars for autoencoder training
