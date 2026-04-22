import random

STOPWORDS = {
    "is","the","and","a","an","in","on","at","to","for","of","with","by","as","it"
}

def perturb_text(text):
    words = text.split()
    n = len(words)

    if n < 6:
        return [text]

    variations = []

    for _ in range(3):
        new_words = words.copy()

        # ---- 1. Replace ~20% words ----
        replace_count = max(1, int(0.2 * n))
        replace_indices = random.sample(range(n), replace_count)

        for i in replace_indices:
            if new_words[i].lower() not in STOPWORDS:
                new_words[i] = random.choice(["very", "really", "quite", "extremely"])

        # ---- 2. Delete ~15% words ----
        delete_count = max(1, int(0.15 * n))
        delete_indices = sorted(random.sample(range(len(new_words)), delete_count), reverse=True)

        for i in delete_indices:
            if new_words[i].lower() not in STOPWORDS:
                del new_words[i]

        variations.append(" ".join(new_words))

    return variations