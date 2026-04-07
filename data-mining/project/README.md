# Data Mining Assignment

Before anything else:

```sh
cd data-mining
```

Run scripts from here like:

```sh
python -m project.src.data --data data
```

## Creating datasets

Running the following command will create the datasets in the `data` directory. The datasets are created from the raw data in the `data` directory, which is not included in this repository. Make sure you have the following files inside the `data` directory:

- `processed_lyrics.tar.gz`: A tarball containing the processed lyrics of the songs.
- `id_genres.csv`: A CSV file containing the genres of the songs.
- `id_mfcc_stats.tsv.bz2`: A TSV file containing the MFCC statistics of the songs.
- `id_information.csv`: A CSV file containing the information of the songs.
- `id_tags.csv`: A CSV file containing the tags of the songs.

```sh
python -m src.data --data data
```

To generate different datasets, for different top $k$ genres, run the following command:

```sh
for i in {1..5}; do python project/src/data.py project/data -k $i; done
```
