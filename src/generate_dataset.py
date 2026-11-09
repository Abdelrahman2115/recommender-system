"""Synthetic dataset generator (script version of notebooks/00_generate_dataset.ipynb).

Run from the project root:  python src/generate_dataset.py
Writes data/raw/users.csv, events.csv and interactions.csv (seed 42 -> identical files every run).
"""
# %% [setup]
# ---------------------------------------------------------------
# Stage 0 - Synthetic dataset generator for the Event Recommendation System
# ---------------------------------------------------------------
import numpy as np            # fast math on arrays (random numbers, vectors)
import pandas as pd           # tables (DataFrames) that we save as CSV
import os                     # create folders

SEED = 42                                  # fixed seed -> same dataset every run (reproducible)
rng = np.random.default_rng(SEED)          # modern NumPy random generator

N_USERS = 2000                             # number of users
N_EVENTS = 800                             # number of events
START = pd.Timestamp("2024-01-01")         # first day of the platform's history
END = pd.Timestamp("2025-12-31")           # last day of the platform's history

OUT_DIR = "data/raw"                       # where the CSV files will be written
os.makedirs(OUT_DIR, exist_ok=True)        # make the folder if it does not exist

# %% [vocab]
# ---------------------------------------------------------------
# 1) The "world": categories, sub-topics, and the words used to describe them
# ---------------------------------------------------------------
# Each category has sub-topics; each sub-topic has its own keywords.
# Keywords make event text meaningful, so TF-IDF has something real to learn.
CATEGORIES = {
    "Technology": {
        "AI & Data":        ["machine learning", "data science", "python", "neural networks", "analytics"],
        "Web & Mobile":     ["javascript", "react", "flutter", "web development", "apps"],
        "Cybersecurity":    ["security", "ethical hacking", "networks", "cloud", "privacy"],
    },
    "Arts & Culture": {
        "Painting":         ["painting", "canvas", "watercolor", "gallery", "sketching"],
        "Theatre & Film":   ["theatre", "cinema", "screening", "acting", "film"],
        "Heritage":         ["museum", "history", "pharaonic", "architecture", "heritage"],
    },
    "Sports & Fitness": {
        "Running":          ["running", "marathon", "5k", "endurance", "track"],
        "Football":         ["football", "five-a-side", "match", "league", "tournament"],
        "Yoga & Wellness":  ["yoga", "meditation", "stretching", "mindfulness", "wellness"],
    },
    "Food & Drink": {
        "Cooking Classes":  ["cooking", "recipes", "chef", "kitchen", "baking"],
        "Food Festivals":   ["street food", "festival", "tasting", "food trucks", "desserts"],
        "Coffee":           ["coffee", "specialty coffee", "barista", "brewing", "roasting"],
    },
    "Music": {
        "Live Concerts":    ["concert", "live band", "stage", "rock", "pop"],
        "Oriental Music":   ["oud", "tarab", "oriental", "classical arabic", "orchestra"],
        "Electronic":       ["dj", "electronic", "techno", "house", "beats"],
    },
    "Business & Career": {
        "Startups":         ["startup", "entrepreneurship", "pitching", "investors", "funding"],
        "Career Growth":    ["career", "cv", "interview skills", "networking", "leadership"],
        "Marketing":        ["marketing", "branding", "social media", "content", "sales"],
    },
    "Education & Workshops": {
        "Languages":        ["english", "german", "french", "language exchange", "conversation"],
        "Science":          ["science", "astronomy", "physics", "lecture", "research"],
        "Photography":      ["photography", "camera", "editing", "portrait", "lightroom"],
    },
    "Outdoors & Travel": {
        "Hiking & Camping": ["hiking", "camping", "desert", "trail", "stars"],
        "Diving & Sea":     ["diving", "snorkeling", "red sea", "beach", "coral"],
        "City Tours":       ["walking tour", "old cairo", "sightseeing", "guided tour", "markets"],
    },
}
CAT_NAMES = list(CATEGORIES.keys())        # the 8 category names in a fixed order
N_CATS = len(CAT_NAMES)

# Egyptian cities with a share of the population (Cairo is the biggest market).
CITIES = ["Cairo", "Giza", "Alexandria", "Mansoura", "Tanta", "Hurghada", "Luxor"]
CITY_P = np.array([0.40, 0.20, 0.18, 0.08, 0.05, 0.05, 0.04])

# Some categories are naturally linked to some cities (diving -> Hurghada, heritage -> Luxor).
CITY_BOOST = {("Outdoors & Travel", "Hurghada"): 3.0, ("Outdoors & Travel", "Luxor"): 1.5,
              ("Arts & Culture", "Luxor"): 2.0, ("Technology", "Cairo"): 1.5,
              ("Business & Career", "Cairo"): 1.5}

VENUES = ["co-working space", "hall", "outdoor", "cafe", "university", "online"]

# Hidden "style" dimensions: things that drive taste but are NOT columns in the data.
#   0: beginner (-) vs advanced (+)
#   1: calm (-)     vs energetic / social (+)
#   2: budget (-)   vs premium (+)
STYLE_DIM = 3
STYLE_WORDS = {  # words that sometimes leak the style into the text (only partially observable)
    (0, -1): ["beginner-friendly", "introductory", "no experience needed"],
    (0, +1): ["advanced", "deep dive", "for professionals"],
    (1, -1): ["relaxed", "small group", "calm atmosphere"],
    (1, +1): ["high energy", "meet new people", "big crowd"],
    (2, -1): ["affordable", "free entry", "student discount"],
    (2, +1): ["premium", "exclusive", "vip"],
}
GENERIC = ["join us", "community", "weekend", "experience", "learn", "fun", "friends",
           "session", "evening", "event", "hosted by", "special", "egypt"]

# %% [events]
# ---------------------------------------------------------------
# 2) Generate EVENTS
# ---------------------------------------------------------------
def make_event_text(cat, sub, style, rng):
    """Build a title and description from the sub-topic keywords + style words + noise."""
    kws = CATEGORIES[cat][sub]
    main = rng.choice(kws)                                   # the main keyword of the event
    title_tpl = rng.choice(["{k} meetup", "{k} workshop", "{k} night", "intro to {k}",
                            "{k} session", "{k} gathering", "{k} masterclass"])
    title = title_tpl.format(k=main).title()

    words = list(rng.choice(kws, size=3, replace=False))     # 3 topic keywords
    # Style words appear only when the hidden style is strong (and only 60% of the time),
    # so the text gives a hint about the hidden taste but never the full truth.
    for d in range(STYLE_DIM):
        if abs(style[d]) > 0.6 and rng.random() < 0.6:
            words.append(rng.choice(STYLE_WORDS[(d, int(np.sign(style[d])))]))
    # 15% chance to borrow a keyword from another category (real events mix topics).
    if rng.random() < 0.15:
        other_cat = rng.choice([c for c in CAT_NAMES if c != cat])
        other_sub = rng.choice(list(CATEGORIES[other_cat].keys()))
        words.append(rng.choice(CATEGORIES[other_cat][other_sub]))
    words += list(rng.choice(GENERIC, size=3, replace=False))  # generic filler words
    rng.shuffle(words)
    desc = f"{title}. A {sub.lower()} event about " + ", ".join(words) + "."
    return title, desc, kws

cat_share = rng.dirichlet(np.full(N_CATS, 5.0))              # some categories have more events
ev_rows, ev_hidden = [], []
for eid in range(1, N_EVENTS + 1):
    cat = CAT_NAMES[rng.choice(N_CATS, p=cat_share)]
    sub = rng.choice(list(CATEGORIES[cat].keys()))
    # city: population share, boosted for natural category-city pairs
    w = np.array([CITY_P[i] * CITY_BOOST.get((cat, c), 1.0) for i, c in enumerate(CITIES)])
    city = rng.choice(CITIES, p=w / w.sum())
    style = rng.normal(0, 1, STYLE_DIM).clip(-2, 2)           # hidden event style
    quality = rng.normal(0, 1)                                # hidden quality -> popularity
    title, desc, kws = make_event_text(cat, sub, style, rng)
    tags = "|".join(rng.choice(kws, size=2, replace=False))
    venue = rng.choice(VENUES, p=[0.15, 0.25, 0.15, 0.2, 0.15, 0.10])
    # price in EGP: free sometimes; otherwise log-normal, higher for "premium" style
    price = 0.0 if rng.random() < 0.25 else round(float(np.exp(rng.normal(5.3 + 0.35 * style[2], 0.6))), -1)
    start = START + pd.Timedelta(days=int(rng.integers(30, (END - START).days)),
                                 hours=int(rng.choice([10, 12, 16, 18, 19, 20])))
    organizer = int(rng.integers(1, 120))
    ev_rows.append(dict(event_id=eid, title=title, description=desc, category=cat, tags=tags,
                        city=city, venue_type=venue, start_datetime=start, price=price,
                        organizer_id=organizer))
    ev_hidden.append(dict(event_id=eid, cat_idx=CAT_NAMES.index(cat), style=style, quality=quality))

events = pd.DataFrame(ev_rows)
E_cat = np.array([h["cat_idx"] for h in ev_hidden])          # hidden arrays used only by the generator
E_style = np.vstack([h["style"] for h in ev_hidden])
E_quality = np.array([h["quality"] for h in ev_hidden])
E_city = events["city"].to_numpy()
E_start = events["start_datetime"].to_numpy()

# %% [users]
# ---------------------------------------------------------------
# 3) Generate USERS (with hidden true tastes)
# ---------------------------------------------------------------
# True category preference: Dirichlet with small alpha -> each user cares about a few categories.
U_pref = rng.dirichlet(np.full(N_CATS, 0.35), size=N_USERS)
U_style = rng.normal(0, 1, (N_USERS, STYLE_DIM))              # hidden style taste
U_city = rng.choice(CITIES, size=N_USERS, p=CITY_P)
U_travel = rng.beta(2, 6, N_USERS)                            # willingness to go to other cities
U_price_sens = rng.gamma(2.0, 0.5, N_USERS)                   # how much price hurts
# Activity follows a long tail: most users do little, a few do a lot.
U_activity = np.clip(np.round(rng.lognormal(2.9, 0.8, N_USERS)), 3, 250).astype(int)
# Rating habit: some people rate generously, others harshly (a classic real-world effect).
# Drawn from a separate generator so it does not change any other part of the data.
U_leniency = np.random.default_rng(SEED + 7).normal(0, 0.45, N_USERS)

def stated_interests(pref, rng):
    """What the user TYPES in their profile: related to true taste but imperfect."""
    order = np.argsort(-pref)
    chosen = [CAT_NAMES[i] for i in order[:3] if pref[i] > 0.12 and rng.random() < 0.85]
    if not chosen:                                            # everyone picks at least one
        chosen = [CAT_NAMES[order[0]]]
    if rng.random() < 0.30:                                   # 30%: add a random "aspirational" interest
        chosen.append(rng.choice(CAT_NAMES))
    return "|".join(dict.fromkeys(chosen))                    # remove duplicates, keep order

users = pd.DataFrame({
    "user_id": np.arange(1, N_USERS + 1),
    "age": np.clip(rng.normal(27, 7, N_USERS).round(), 16, 65).astype(int),
    "city": U_city,
    "interests": [stated_interests(p, rng) for p in U_pref],
    # signup spread over the period, with more signups later (the platform grows)
    "signup_date": START + pd.to_timedelta((rng.beta(1.3, 1.0, N_USERS) * 600).astype(int), unit="D"),
})

# %% [interactions]
# ---------------------------------------------------------------
# 4) Generate INTERACTIONS from the hidden tastes
# ---------------------------------------------------------------
price_norm = np.log1p(events["price"].to_numpy()) / np.log1p(events["price"].max())

def affinity(u):
    """Hidden 'true' score of every event for user u (the thing a recommender tries to learn)."""
    cat_part = 2.2 * np.log(U_pref[u, E_cat] + 0.02)          # category taste (strongest signal)
    style_part = 0.55 * (E_style @ U_style[u])                # hidden style match (CF must discover it)
    qual_part = 0.6 * E_quality                               # good events attract everyone
    same_city = (E_city == U_city[u])
    city_part = np.where(same_city, 0.0, np.log(U_travel[u] + 0.02))  # other cities are less likely
    price_part = -1.2 * U_price_sens[u] * price_norm          # expensive events hurt price-sensitive users
    online = (events["venue_type"].to_numpy() == "online")
    city_part = np.where(online, 0.0, city_part)              # online events ignore distance
    return cat_part + style_part + qual_part + city_part + price_part

rows = []
for u in range(N_USERS):
    signup = users.loc[u, "signup_date"]
    # a user can only see events that start after they joined
    available = np.where(E_start > np.datetime64(signup + pd.Timedelta(days=3)))[0]
    if len(available) < 5:
        continue
    aff = affinity(u)[available]
    # exposure: the user browses events with probability ~ softmax(affinity) + random noise
    noisy = aff + rng.gumbel(0, 1.0, len(available))          # Gumbel noise = sampling from softmax
    n = min(U_activity[u], len(available))
    picked_local = np.argsort(-noisy)[:n]                     # top-n noisy scores = sampled events
    pa = aff[picked_local]                                    # affinities of the events this user browsed
    for j in picked_local:
        e = available[j]
        # funnel: compared with the OTHER events this user browsed, a better fit -> deeper engagement
        z = (aff[j] - pa.mean()) / (pa.std() + 1e-9)
        p_going = 1 / (1 + np.exp(-1.6 * (z - 0.9)))          # only clearly good fits get "going"
        p_interested = 1 / (1 + np.exp(-1.6 * (z - 0.1)))
        p_attend = 0.80                                       # 80% of "going" actually attend
        if rng.random() < p_going:
            itype = "attended" if rng.random() < p_attend else "going"
        elif rng.random() < p_interested:
            itype = "interested"
        else:
            itype = "view"
        start = pd.Timestamp(E_start[e])
        if itype == "attended":
            ts = start + pd.Timedelta(hours=int(rng.integers(3, 72)))   # rated after attending
            # rating = user's rating habit + event quality + how well it fits this user + noise
            rating = int(np.clip(np.round(2.8 + U_leniency[u] + 0.45 * E_quality[e] + 0.6 * z
                                          + rng.normal(0, 0.6)), 1, 5))
        else:
            lead = int(rng.integers(1, 25))
            ts = max(start - pd.Timedelta(days=lead), signup + pd.Timedelta(hours=1))
            ts += pd.Timedelta(minutes=int(rng.integers(0, 1440)))
            rating = np.nan                                   # no rating if not attended
        rows.append((u + 1, e + 1, ts, itype, rating))

interactions = pd.DataFrame(rows, columns=["user_id", "event_id", "timestamp",
                                           "interaction_type", "rating"])
interactions = interactions[interactions["timestamp"] <= END].sort_values("timestamp").reset_index(drop=True)

# %% [mess]
# ---------------------------------------------------------------
# 5) Add realistic data-quality problems (fixed later in Stage 4)
# ---------------------------------------------------------------
# Real data is never clean. We add a SMALL, documented amount of mess:
mess_rng = np.random.default_rng(SEED + 1)

# a) 1% duplicated interaction rows (e.g., a double click logged twice)
dup = interactions.sample(frac=0.01, random_state=SEED)
interactions = pd.concat([interactions, dup]).sort_values("timestamp").reset_index(drop=True)

# b) 3% missing ages and a few impossible ages (typing mistakes)
users["age"] = users["age"].astype("float")
users.loc[users.sample(frac=0.03, random_state=SEED).index, "age"] = np.nan
users.loc[users.sample(n=8, random_state=SEED + 2).index, "age"] = [3, 150, 999, 0, 7, 120, 210, 1]

# c) inconsistent city spelling / spacing in a few rows
bad = users.sample(frac=0.02, random_state=SEED + 3).index
users.loc[bad, "city"] = users.loc[bad, "city"].map(lambda c: mess_rng.choice([c.lower(), c.upper(), f" {c} "]))

# d) a few events with missing description or missing price
events.loc[events.sample(n=12, random_state=SEED).index, "description"] = np.nan
events.loc[events.sample(n=10, random_state=SEED + 4).index, "price"] = np.nan

# %% [save]
# ---------------------------------------------------------------
# 6) Save the three CSV files (hidden tastes are NOT saved -> no leakage)
# ---------------------------------------------------------------
users.to_csv(f"{OUT_DIR}/users.csv", index=False)
events.to_csv(f"{OUT_DIR}/events.csv", index=False)
interactions.to_csv(f"{OUT_DIR}/interactions.csv", index=False)
print("users:", users.shape, "| events:", events.shape, "| interactions:", interactions.shape)
print(interactions["interaction_type"].value_counts())
