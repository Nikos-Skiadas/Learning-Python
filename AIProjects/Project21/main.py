from __future__ import annotations

# Import necessary libraries
import os
import re
import string
import typing
import warnings; warnings.filterwarnings("ignore", category=UserWarning)  # Ignore warnings

import numpy
from rich import print  # For colorful print statements
import pandas

# Importing Scikit-learn for machine learning
import sklearn
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.pipeline
import sklearn.metrics
import sklearn.model_selection

# Importing NLTK for Natural Language Processing
import nltk
import nltk.stem
import nltk.corpus

# Download required NLP datasets
nltk.download('wordnet')  # For lemmatization
nltk.download('stopwords')  # For removing stopwords

# Function to load dataset (train, val, test)
def load_data(
    split: typing.Literal["train", "val", "test"],
    root: str = "",
    index: str = "ID",
):
    return pandas.read_csv(os.path.join(f"{root}", f"{split}_dataset.csv"),
        index_col=index,  # Set ID column as index
        encoding="utf-8",
    )

# Class for text preprocessing (cleaning the text)
class Preprocessor:
    def __call__(self, text: str) -> str:
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"#\w+", "", text)  # Remove hashtags
        text = re.sub(r"\S+@\S+", "", text)  # Remove emails
        return text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation

# Class for tokenization and stemming/lemmatization
class Tokenizer:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()
        self.tokenizer = nltk.tokenize.TweetTokenizer(
            preserve_case=False,
            reduce_len=True,
            match_phone_numbers=True,
            strip_handles=True,
        )

    def __call__(self, text: str):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(token))
            for token in self.tokenizer.tokenize(text) if token and not token.isdigit()]

# Class to convert text into numerical features using TF-IDF
class Vectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def __init__(self, *, max_features: int | None = None, ngram_range: tuple[int, int] = (1, 1)):
        super().__init__(
            encoding="utf-8",
            strip_accents="unicode",
            lowercase=True,
            preprocessor=Preprocessor(),
            tokenizer=Tokenizer(),
            stop_words=nltk.corpus.stopwords.words("english"),  # Remove stopwords
            ngram_range=ngram_range,  # Define n-gram range
            max_features=max_features,  # Limit feature size to avoid overfitting
        )

# Logistic Regression model for classification
class Model(sklearn.linear_model.LogisticRegression):
    def __init__(self, *, C: float = 1.0):
        super().__init__(
            C=C,  # Regularization parameter (can be tuned)
            fit_intercept=True,  # Include bias term
            random_state=42,  # Ensure reproducibility
            n_jobs=-1,  # Use all CPU cores for parallel processing
        )

# Classifier that combines vectorization and model into a pipeline
class Classifier:
    def __init__(self, *, vectorizer: Vectorizer, model: Model):
        self.pipeline = sklearn.pipeline.Pipeline([
            ("vectorizer", vectorizer),
            ("model", model),
        ])

    # Train the classifier
    def fit(self, train_data: pandas.DataFrame):
        X = train_data.Text
        y = train_data.Label
        self.pipeline.fit(X, y)
        return self

    # Make predictions on new data
    def predict(self, test_data: pandas.DataFrame, save: str | None = None) -> pandas.Series:
        X = test_data.Text
        y = pandas.Series(self.pipeline.predict(X),
            index=test_data.index,
            name="Label",
        )
        if save is not None:
            y.to_csv(save)  # Save predictions to CSV
        return y

    # Evaluate the model using accuracy, precision, recall, and F1-score
    def evaluate(self, val_data: pandas.DataFrame):
        y_true = val_data.Label
        y_pred = self.predict(val_data)
        return {
            "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
            "precision": sklearn.metrics.precision_score(y_true, y_pred),
            "recall": sklearn.metrics.recall_score(y_true, y_pred),
            "f1": sklearn.metrics.f1_score(y_true, y_pred),
        }

    # Hyperparameter tuning using GridSearchCV
    def tune(self, train_data: pandas.DataFrame, val_data: pandas.DataFrame, **param_grid: list):
        data = pandas.concat([train_data, val_data])  # Combine training and validation sets
        X = data.Text
        y = data.Label

        tuner = sklearn.model_selection.GridSearchCV(
            self.pipeline, param_grid,
            scoring="accuracy",
            n_jobs=-1,  # Use all available CPU cores
            refit=False,  # We will refit manually
            cv=sklearn.model_selection.PredefinedSplit([-1] * len(train_data) + [0] * len(val_data)),
        )

        tuner.fit(X, y)  # type: ignore
        self.pipeline.set_params(**tuner.best_params_)  # Apply best parameters
        self.pipeline.fit(X, y)  # Refit model with optimal hyperparameters
        return self

# Main execution
if __name__ == "__main__":
    # Load datasets
    train_data = load_data("train")
    val_data = load_data("val")
    test_data = load_data("test")

    # Create classifier instance
    classifier = Classifier(
        vectorizer=Vectorizer(),
        model=Model(),
    )

    # Perform hyperparameter tuning
    classifier.tune(train_data, val_data,
        vectorizer__max_features=[128, 256, 512],
        vectorizer__ngram_range=[(1, 1), (1, 2), (2, 2)],
        model__C=list(numpy.linspace(.5, 1., 2)),
    )

    # Generate predictions and save to submission.csv
    predictions = classifier.predict(test_data, save="submission.csv")
