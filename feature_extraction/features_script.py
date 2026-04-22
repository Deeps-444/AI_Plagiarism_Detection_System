import numpy as np
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
# ---------------- TEXT PROCESSING ----------------



def sentence_split(text):
    return sent_tokenize(text)

def word_tokenize(text):
    return re.findall(r'\w+', text.lower())

def avg_word_length(text):
    words = word_tokenize(text)
    return np.mean([len(w) for w in words]) if words else 0

def sentence_length_variance(text):
    sentences = sentence_split(text)
    lengths = [len(word_tokenize(s)) for s in sentences if s.strip()]
    return np.var(lengths) if len(lengths) >= 2 else 0

def avg_sentence_length(text):
    sentences = sentence_split(text)
    lengths = [len(word_tokenize(s)) for s in sentences if s.strip()]
    return np.mean(lengths) if lengths else 0

def punctuation_ratio(text):
    punct = len(re.findall(r'[^\w\s]', text))
    return punct / len(text) if len(text) > 0 else 0

def lexical_diversity(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words) if words else 0


# ---------------- MODELS ----------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------------- FEATURES ----------------

def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = gpt2_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def embedding_variance(text):
    sentences = [s.strip() for s in sentence_split(text) if s.strip()]

    if len(sentences) < 2:
        return 0.0

    embeddings = embedding_model.encode(sentences)

    # pairwise semantic distance (STRONG signal)
    distances = cosine_distances(embeddings)

    # take upper triangle (avoid duplicates + self-comparison)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]

    return float(np.mean(upper_triangle))


# ---------------- STABILITY ----------------

def stability_score(predict_fn, text, perturb_fn):
    variations = perturb_fn(text)
    preds = [predict_fn(v) for v in variations]
    return np.var(preds)



def extract_features(text):
    return [
        compute_perplexity(text),
        sentence_length_variance(text),
        embedding_variance(text),
        avg_word_length(text),
        punctuation_ratio(text),
        lexical_diversity(text),
        avg_sentence_length(text)
    ]
