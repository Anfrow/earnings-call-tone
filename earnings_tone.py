"""
Earnings Call Tone Analyzer - Step 1
Scores the sentiment of one earnings call transcript with FinBERT.

Setup (run once in your terminal):
    pip install transformers torch

Usage:
    1. Paste a transcript into a file called transcript.txt (same folder as this script)
    2. Run:  python earnings_tone.py
"""

from transformers import pipeline

# 1. Load FinBERT, a model trained on financial text.
#    It labels text as "positive", "negative", or "neutral".
#    The first run downloads the model (~400 MB), so it'll be slow once.
finbert = pipeline("text-classification", model="ProsusAI/finbert")

# 2. Read the transcript.
with open("transcript.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. Split into paragraphs. FinBERT can only read ~512 tokens at a time,
#    so we score small pieces instead of the whole call at once.
paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
print(f"Scoring {len(paragraphs)} paragraphs...\n")

# 4. Score every paragraph.
#    truncation=True cuts off anything too long so it doesn't crash.
results = finbert(paragraphs, truncation=True)

# 5. Turn labels into numbers: positive = +1, neutral = 0, negative = -1,
#    weighted by how confident the model is.
label_value = {"positive": 1, "neutral": 0, "negative": -1}
scores = [label_value[r["label"]] * r["score"] for r in results]

# 6. Overall tone = average of all paragraph scores (-1 to +1).
overall = sum(scores) / len(scores)

counts = {"positive": 0, "neutral": 0, "negative": 0}
for r in results:
    counts[r["label"]] += 1

print(f"Positive paragraphs: {counts['positive']}")
print(f"Neutral paragraphs:  {counts['neutral']}")
print(f"Negative paragraphs: {counts['negative']}")
print(f"\nOverall tone score: {overall:+.3f}  (-1 = very negative, +1 = very positive)")

# 7. Show the 3 most negative paragraphs - often the most interesting part.
ranked = sorted(zip(scores, paragraphs))
print("\nMost negative moments:")
for score, para in ranked[:3]:
    print(f"  [{score:+.2f}] {para[:150]}...")
