"""Turn events into numeric vectors (explained in notebooks/04_feature_engineering.ipynb)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

PRICE_BINS = [-1, 0, 150, 400, np.inf]                # free | cheap | medium | expensive (EGP)
PRICE_LABELS = ["free", "cheap", "medium", "expensive"]


def event_text(events):
    """Combine every text field of an event into one document (a 'soup' of words)."""
    return (events["category"] + " " + events["tags"].str.replace("|", " ", regex=False) + " "
            + events["title"] + " " + events["description"]).str.lower()


def price_bucket(events):
    """Group prices into 4 buckets so 'cheap' events are similar to each other."""
    return pd.cut(events["price"], bins=PRICE_BINS, labels=PRICE_LABELS).astype(str)


class EventFeaturizer:
    """Builds one vector per event from 4 blocks: text (TF-IDF), category, city and price.

    Each block is L2-normalised and multiplied by sqrt(weight). Because of this, the cosine
    similarity of two events equals the weighted average of the per-block similarities:
        cos = (w_text*cos_text + w_cat*cos_cat + w_city*cos_city + w_price*cos_price) / sum(w)
    which makes the weights easy to explain and tune.
    """

    def __init__(self, text_weight=1.0, category_weight=0.5, city_weight=0.5, price_weight=0.2,
                 ngram_range=(1, 2), min_df=2):
        self.weights = {"text": text_weight, "category": category_weight,
                        "city": city_weight, "price": price_weight}
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=ngram_range,
                                     min_df=min_df, sublinear_tf=True)
        self.onehot = {name: OneHotEncoder(handle_unknown="ignore") for name in ("category", "city", "price")}

    def _blocks(self, events, fit):
        cols = {"category": events[["category"]], "city": events[["city"]],
                "price": price_bucket(events).to_frame("price")}
        text = event_text(events)
        blocks = {"text": self.tfidf.fit_transform(text) if fit else self.tfidf.transform(text)}
        for name, frame in cols.items():
            enc = self.onehot[name]
            blocks[name] = enc.fit_transform(frame) if fit else enc.transform(frame)
        return blocks

    def _combine(self, blocks):
        parts = [normalize(blocks[n]) * np.sqrt(self.weights[n]) for n in ("text", "category", "city", "price")]
        return normalize(csr_matrix(hstack(parts)))

    def fit_transform(self, events):
        return self._combine(self._blocks(events, fit=True))

    def transform(self, events):
        return self._combine(self._blocks(events, fit=False))

    def city_vector(self, city):
        """A vector that only has the 'city' block switched on (used for a user's home city)."""
        sizes = [len(self.tfidf.vocabulary_)] + [len(self.onehot[n].categories_[0])
                                                  for n in ("category", "city", "price")]
        vec = np.zeros(sum(sizes))
        city_block = self.onehot["city"].transform(pd.DataFrame({"city": [city]})).toarray()[0]
        start = sizes[0] + sizes[1]                      # the city block comes after text + category
        vec[start:start + sizes[2]] = city_block
        return vec
