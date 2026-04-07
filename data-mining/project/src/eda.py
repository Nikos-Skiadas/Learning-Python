from __future__ import annotations


import argparse
import pathlib
from dataclasses import dataclass

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot
import numpy
import pandas
import seaborn
import torch

from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from wordcloud import WordCloud


@dataclass
class EDAData:
	dataset: pandas.DataFrame
	genres_enc: pandas.DataFrame
	tags_enc: pandas.DataFrame
	audio_emb: pandas.DataFrame
	lyrics_emb: pandas.DataFrame


def load_eda_data(data_dir: pathlib.Path, k: int = 5) -> EDAData:
	return EDAData(
		dataset = pandas.read_csv(data_dir / f"dataset.{k}.csv", index_col = 0),
		genres_enc = pandas.read_parquet(data_dir / f"dataset.{k}.genres.parquet"),
		tags_enc = pandas.read_parquet(data_dir / f"dataset.{k}.tags.parquet"),
		audio_emb = pandas.read_parquet(data_dir / f"dataset.{k}.audio.parquet"),
		lyrics_emb = pandas.read_parquet(data_dir / f"dataset.{k}.lyrics.parquet"),
	)


def top_genre_names(data: EDAData, k: int = 5) -> list[str]:
	return data.genres_enc.sum().sort_values(ascending = False).head(k).index.tolist()


def most_different_genres(data: EDAData, genres: list[str]) -> tuple[str, str]:
	profiles = {}

	for genre in genres:
		mask = data.genres_enc[genre].astype(bool)
		profiles[genre] = data.tags_enc.loc[mask].sum().values.astype(float)
		profiles[genre] /= profiles[genre].sum() or 1.

	worst_sim = 1.
	pair = (genres[0], genres[1])

	for i, g1 in enumerate(genres):
		for g2 in genres[i + 1:]:
			sim = cosine_similarity(
				profiles[g1].reshape(1, -1),
				profiles[g2].reshape(1, -1),
			)[0, 0]

			if sim < worst_sim:
				worst_sim = sim
				pair = (g1, g2)

	return pair


def plot_wordcloud(data: EDAData, genre: str,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	mask = data.genres_enc[genre].astype(bool)
	tag_freq = data.tags_enc.loc[mask].sum().to_dict()

	cloud = WordCloud(
		width = 800, height = 400,
		background_color = "white",
	).generate_from_frequencies(tag_freq)

	if ax is None: fig, ax = matplotlib.pyplot.subplots(figsize = (10, 5))
	else: fig = ax.figure

	ax.imshow(cloud, interpolation = "bilinear")
	ax.set_title(f"Tag Word Cloud: {genre}")
	ax.axis("off")

	return fig  # type: ignore


def plot_top_tags(data: EDAData, n: int = 10,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	counts = data.tags_enc.sum().sort_values(ascending = True).tail(n)

	if ax is None: fig, ax = matplotlib.pyplot.subplots(figsize = (10, 6))
	else: fig = ax.figure

	ax.barh(counts.index, counts.values)  # type: ignore
	ax.set_xlabel("Frequency")
	ax.set_title(f"Top {n} Most Frequent Tags")

	return fig  # type: ignore


def reduce_tsne(embeddings: pandas.DataFrame,
	n_components: int = 2,
	**kwargs,
) -> numpy.ndarray:
	return TSNE(
		n_components = n_components,
		random_state = 42,
		**kwargs,
	).fit_transform(embeddings.values)


def plot_scatter_by_genre(data: EDAData,
	modality: str,
	coords_2d: numpy.ndarray,
	top_k: int = 5,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	if ax is None: fig, ax = matplotlib.pyplot.subplots(figsize = (10, 8))
	else: fig = ax.figure

	genres = top_genre_names(data, top_k)

	for genre in genres:
		mask = data.genres_enc[genre].astype(bool).values
		ax.scatter(
			coords_2d[mask, 0],  # type: ignore
			coords_2d[mask, 1],  # type: ignore
			label = genre, alpha = 0.1, s = 5,
		)

	ax.set_title(f"t-SNE: {modality.capitalize()} Embeddings by Genre")
	ax.legend(markerscale = 5)

	return fig  # type: ignore


def plot_modality_comparison(data: EDAData, top_k: int = 5) -> matplotlib.figure.Figure:
	audio_2d = reduce_tsne(data.audio_emb)
	lyrics_2d = reduce_tsne(data.lyrics_emb)

	fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize = (20, 8))

	plot_scatter_by_genre(data, "audio", audio_2d, top_k, ax = ax1)
	plot_scatter_by_genre(data, "lyrics", lyrics_2d, top_k, ax = ax2)

	fig.suptitle("Audio vs Text Embeddings — Genre Separation Comparison")
	fig.tight_layout()

	return fig


def plot_genre_count_histogram(data: EDAData,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	genre_counts = data.dataset["genres"].str.count(",") + 1

	if ax is None: fig, ax = matplotlib.pyplot.subplots(figsize = (10, 6))
	else: fig = ax.figure

	bins = range(1, genre_counts.max() + 2)

	ax.hist(genre_counts, bins = bins, edgecolor = "black", align = "left")
	ax.set_xlabel("Number of Genres per Song")
	ax.set_ylabel("Number of Songs")
	ax.set_title("Genre Variety: How Many Genres Does Each Song Belong To?")
	ax.set_xticks(range(1, genre_counts.max() + 1))

	return fig  # type: ignore


def plot_genre_distribution(data: EDAData, n: int = 10,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	counts = data.genres_enc.sum().sort_values(ascending = True).tail(n)

	if ax is None: fig, ax = matplotlib.pyplot.subplots(figsize = (10, 6))
	else: fig = ax.figure

	ax.barh(counts.index, counts.values)  # type: ignore
	ax.set_xlabel("Number of Songs")
	ax.set_title(f"Top {n} Genres by Song Count")

	return fig  # type: ignore


def plot_lyrics_length(data: EDAData,
	lang: str = "english",
) -> matplotlib.figure.Figure:
	lyrics = data.dataset["lyrics"]
	words = lyrics.str.split()

	stop = set(stopwords.words(lang))

	char_counts = lyrics.str.len()
	word_counts = words.str.len()
	meaningful_counts = words.apply(lambda ws: sum(1 for w in ws if w not in stop))

	fig, axes = matplotlib.pyplot.subplots(1, 3, figsize = (18, 5))

	axes[0].hist(char_counts, bins = 50, edgecolor = "black")
	axes[0].set_xlabel("Character Count")
	axes[0].set_ylabel("Number of Songs")
	axes[0].set_title("Lyrics Length (Characters)")

	axes[1].hist(word_counts, bins = 50, edgecolor = "black")
	axes[1].set_xlabel("Word Count")
	axes[1].set_title("Lyrics Length (Words)")

	axes[2].hist(meaningful_counts, bins = 50, edgecolor = "black")
	axes[2].set_xlabel("Meaningful Word Count")
	axes[2].set_title("Lyrics Length (Without Stopwords)")

	fig.tight_layout()

	return fig


def compute_sentiment(lyrics: pandas.Series,
	batch_size: int = 64,
) -> pandas.Series:
	device = 0 if torch.cuda.is_available() else -1

	classifier = pipeline("sentiment-analysis",  # type: ignore
		device = device,
		truncation = True,
		max_length = 512,
	)

	results = classifier(lyrics.tolist(), batch_size = batch_size)

	scores = pandas.Series(
		[r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results],
		index = lyrics.index,
		name = "sentiment",
	)

	return scores


def plot_sentiment_by_genre(data: EDAData,
	top_k: int = 5,
) -> matplotlib.figure.Figure:
	sentiment = compute_sentiment(data.dataset["lyrics"])
	genres = top_genre_names(data, top_k)

	rows = []

	for genre in genres:
		mask = data.genres_enc[genre].astype(bool)

		for score in sentiment.loc[mask]:
			rows.append({"genre": genre, "sentiment": score})

	expanded = pandas.DataFrame(rows)

	fig, ax = matplotlib.pyplot.subplots(figsize = (10, 6))

	seaborn.violinplot(data = expanded, x = "genre", y = "sentiment", ax = ax)

	ax.set_xlabel("Genre")
	ax.set_ylabel("Sentiment Score")
	ax.set_title("Sentiment Distribution by Genre (DistilBERT)")
	ax.axhline(0, color = "gray", linestyle = "--", alpha = 0.5)

	return fig


def find_similar_songs(data: EDAData, song_id: str,
	k: int = 5,
	modality: str = "both",
) -> pandas.DataFrame:
	results = {}

	if modality in ("lyrics", "both"):
		query = data.lyrics_emb.loc[[song_id]]
		sims = cosine_similarity(query, data.lyrics_emb)[0]
		sims = pandas.Series(sims, index = data.lyrics_emb.index, name = "lyrics_sim")
		sims = sims.drop(song_id).sort_values(ascending = False).head(k)
		results["lyrics_sim"] = sims

	if modality in ("audio", "both"):
		query = data.audio_emb.loc[[song_id]]
		sims = cosine_similarity(query, data.audio_emb)[0]
		sims = pandas.Series(sims, index = data.audio_emb.index, name = "audio_sim")
		sims = sims.drop(song_id).sort_values(ascending = False).head(k)
		results["audio_sim"] = sims

	all_ids = set()

	for s in results.values():
		all_ids.update(s.index)

	info = data.dataset.loc[list(all_ids), ["song", "artist", "genres"]]

	for name, sims in results.items():
		info[name] = sims

	return info.sort_values(
		by = list(results.keys()),
		ascending = False,
	)


def display_similarity_results(data: EDAData, song_id: str,
	k: int = 5,
) -> None:
	song = data.dataset.loc[song_id]

	print(f"\nQuery: {song['artist']} — {song['song']} [{song['genres']}]\n")

	for modality in ("lyrics", "audio"):
		result = find_similar_songs(data, song_id, k, modality)

		print(f"Top {k} by {modality} similarity:")
		print(result[["song", "artist", "genres", f"{modality}_sim"]].to_string())
		print()


# === Orchestrator ===

def run_all(data_dir: pathlib.Path, k: int = 5,
	output_dir: pathlib.Path | None = None,
) -> None:
	data = load_eda_data(data_dir, k)

	def save_or_show(fig: matplotlib.figure.Figure, name: str) -> None:
		if output_dir:
			output_dir.mkdir(parents = True, exist_ok = True)
			fig.savefig(output_dir / f"{name}.png", dpi = 150, bbox_inches = "tight")
			matplotlib.pyplot.close(fig)

		else:
			matplotlib.pyplot.show()

	top_genres = top_genre_names(data, k)
	g1, g2 = most_different_genres(data, top_genres)

	print(f"Top {k} genres: {top_genres}")
	print(f"Most different pair: {g1} vs {g2}")

	fig = plot_wordcloud(data, g1); save_or_show(fig, f"wordcloud_{g1}")
	fig = plot_wordcloud(data, g2); save_or_show(fig, f"wordcloud_{g2}")
	fig = plot_top_tags(data); save_or_show(fig, "top_tags")
	fig = plot_modality_comparison(data, top_k = k); save_or_show(fig, "tsne_comparison")
	fig = plot_genre_count_histogram(data); save_or_show(fig, "genre_count_histogram")
	fig = plot_genre_distribution(data, n = 10); save_or_show(fig, "genre_distribution")
	fig = plot_lyrics_length(data); save_or_show(fig, "lyrics_length")
	fig = plot_sentiment_by_genre(data, top_k = k); save_or_show(fig, "sentiment_by_genre")

	sample_id = data.dataset.index[0]
	display_similarity_results(data, sample_id)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "EDA visualizations for music dataset")
	parser.add_argument("data", type = str, help = "Path to the data directory.")
	parser.add_argument("-k", type = int, default = 5, help = "Number of top genres.")
	parser.add_argument("--output", type = str, default = None, help = "Output directory for figures.")

	args = parser.parse_args()

	run_all(
		pathlib.Path(args.data), args.k,
		pathlib.Path(args.output) if args.output else None,
	)
