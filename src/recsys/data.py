"""Loading, cleaning and time-based splitting of the event dataset.

Every function here is explained step by step in notebooks/03_cleaning_and_split.ipynb.
"""
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"

# How strongly each interaction type signals interest (implicit feedback).
ENGAGEMENT_WEIGHT = {"view": 1, "interested": 2, "going": 3, "attended": 4}

# Interactions that count as a "relevant" (successful) recommendation in ranking metrics.
POSITIVE_TYPES = ("going", "attended")

# Global time cut-offs for the split (see notebook 03 for why the split is global, not per user).
TRAIN_END = pd.Timestamp("2025-07-01")   # train  = everything up to this moment
VAL_END = pd.Timestamp("2025-09-01")     # validation = (TRAIN_END, VAL_END]
DATA_END = pd.Timestamp("2025-12-31 23:59:59")  # test = (VAL_END, DATA_END]

VALID_AGE = (13, 90)                     # ages outside this range are typing mistakes


def load_raw(raw_dir=RAW_DIR):
    """Read the three CSV files and parse their date columns."""
    users = pd.read_csv(f"{raw_dir}/users.csv", parse_dates=["signup_date"])
    events = pd.read_csv(f"{raw_dir}/events.csv", parse_dates=["start_datetime"])
    interactions = pd.read_csv(f"{raw_dir}/interactions.csv", parse_dates=["timestamp"])
    return users, events, interactions


def clean_users(users):
    """Fix city spelling, treat impossible ages as missing, fill missing ages with the median."""
    u = users.copy()
    u["city"] = u["city"].str.strip().str.title()
    bad_age = (u["age"] < VALID_AGE[0]) | (u["age"] > VALID_AGE[1])
    u.loc[bad_age, "age"] = np.nan
    u["age"] = u["age"].fillna(u["age"].median()).astype(int)
    u["interests"] = u["interests"].fillna("")
    return u


def clean_events(events):
    """Fill a missing description with the title, and a missing price with its category's median."""
    e = events.copy()
    e["description"] = e["description"].fillna(e["title"])
    e["price"] = e["price"].fillna(e.groupby("category")["price"].transform("median"))
    e["tags"] = e["tags"].fillna("")
    return e


def clean_interactions(interactions, users, events):
    """Remove duplicates and invalid rows; keep one row per (user, event); add the engagement weight."""
    i = interactions.drop_duplicates().copy()
    i = i[i["user_id"].isin(users["user_id"]) & i["event_id"].isin(events["event_id"])]
    i.loc[i["interaction_type"] != "attended", "rating"] = np.nan      # only attendees can rate
    i = i[i["rating"].isna() | i["rating"].between(1, 5)]
    i = i.sort_values("timestamp").drop_duplicates(["user_id", "event_id"], keep="last")
    i["weight"] = i["interaction_type"].map(ENGAGEMENT_WEIGHT)
    return i.reset_index(drop=True)


def load_clean(raw_dir=RAW_DIR):
    """One call that returns the cleaned users, events and interactions."""
    users, events, interactions = load_raw(raw_dir)
    users = clean_users(users)
    events = clean_events(events)
    interactions = clean_interactions(interactions, users, events)
    return users, events, interactions


def temporal_split(interactions, train_end=TRAIN_END, val_end=VAL_END):
    """Global time split: the past trains the model, the future evaluates it."""
    t = interactions["timestamp"]
    train = interactions[t <= train_end].reset_index(drop=True)
    val = interactions[(t > train_end) & (t <= val_end)].reset_index(drop=True)
    test = interactions[t > val_end].reset_index(drop=True)
    return train, val, test
