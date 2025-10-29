import csv
import pandas as pd
import unicodedata
import string
from itertools import combinations
from rapidfuzz.fuzz import ratio as sim
import os
import numpy as np
import datetime

# =============== Settings ==============='
input_csv = r"C:\Users\akki0\Downloads\projects\SDMO2025Project\project1devs\devs (1).csv"# path to devs.csv
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project1devs")
os.makedirs(output_dir, exist_ok=True)

MIN_PAIRS = 500
MAX_PAIRS = 1000
STEP = 0.01  # threshold adjustment step


# =============== Load developers ===============
DEVS = []
with open(input_csv, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    DEVS = [row for row in reader][1:]  # Skip header


# =============== Normalization ===============
def normalize_text(s):
    if not s:
        return ""
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = unicodedata.normalize("NFKD", s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    s = s.casefold()
    return " ".join(s.split())


def process(dev):
    name_raw, email_raw = dev[0], dev[1]
    name = normalize_text(name_raw)
    email = email_raw.strip().lower()
    prefix = email.split("@")[0] if "@" in email else email

    parts = name.split()
    first = parts[0] if len(parts) >= 1 else ""
    last = parts[-1] if len(parts) > 1 else ""
    i_first = first[0] if first else ""
    return name, first, last, i_first, email, prefix


# =============== Pairwise Bird similarity ===============
rows = []
for a, b in combinations(DEVS, 2):
    name_a, first_a, last_a, i_first_a, email_a, prefix_a = process(a)
    name_b, first_b, last_b, i_first_b, email_b, prefix_b = process(b)

    c1 = sim(name_a, name_b) / 100.0
    c2 = sim(prefix_a, prefix_b) / 100.0
    c31 = sim(first_a, first_b) / 100.0
    c32 = sim(last_a, last_b) / 100.0
    c3 = (c31 + c32) / 2
    c4 = (i_first_a in prefix_b and last_a in prefix_b) or (i_first_b in prefix_a and last_b in prefix_a)

    rows.append([a[0], email_a, b[0], email_b, c1, c2, c3, c4])

df = pd.DataFrame(rows, columns=["name_1", "email_1", "name_2", "email_2", "c1", "c2", "c3", "c4"])

# Save raw similarities (optional)
df.to_csv(os.path.join(output_dir, "devs_similarity_raw.csv"), index=False)

# =============== Threshold adjustment with stricter conditions ===============
threshold = 0.99
selected = pd.DataFrame()

while threshold >= 0.3:
    condition_c1 = df["c1"] >= threshold
    condition_c2 = df["c2"] >= threshold
    condition_c3 = df["c3"] >= threshold
    condition_c4 = df["c4"]

    count_true = condition_c1.astype(int) + condition_c2.astype(int) + condition_c3.astype(int) + condition_c4.astype(int)
    filter_mask = count_true >= 2
    filtered = df[filter_mask]
    count = len(filtered)

    print(f"Threshold {threshold:.3f}: {count} pairs with at least 2 conditions")

    if MIN_PAIRS <= count <= MAX_PAIRS:
        selected = filtered
        print(f"✅ Selected threshold {threshold:.3f} with {count} pairs")
        break

    threshold -= STEP

# Fallback if no threshold satisfies conditions
if selected.empty:
    counts = []
    for t in np.arange(0.3, 0.99, STEP):
        cond_c1 = df["c1"] >= t
        cond_c2 = df["c2"] >= t
        cond_c3 = df["c3"] >= t
        count_c4 = df["c4"]
        ct_true = cond_c1.astype(int) + cond_c2.astype(int) + cond_c3.astype(int) + count_c4.astype(int)
        sel = df[ct_true >= 2]
        counts.append((t, len(sel)))

    best_t, best_count = min(counts, key=lambda x: abs(x[1] - (MIN_PAIRS + MAX_PAIRS) / 2))
    condition_c1 = df["c1"] >= best_t
    condition_c2 = df["c2"] >= best_t
    condition_c3 = df["c3"] >= best_t
    condition_c4 = df["c4"]
    count_true = condition_c1.astype(int) + condition_c2.astype(int) + condition_c3.astype(int) + condition_c4.astype(int)
    selected = df[count_true >= 2]
    threshold = best_t
    print(f"⚠ Fallback: Selected threshold {best_t:.3f} with {best_count} pairs")

# Save output file with timestamp, all pairs meeting criteria
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join(output_dir, f"bird_pairs_t={threshold:.3f}_{timestamp}.csv")
selected.to_csv(file_path, index=False)

print(f"\n✅ Saved filtered pairs to {file_path}")
print(f"✅ Total pairs selected: {len(selected)}")
