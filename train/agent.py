from predictor import predict_fn
from perturbation import perturb_text
from feature_extraction.features_script import extract_features, stability_score


def agent_decision_explain(text):
    feats = extract_features(text)

    perplexity = feats[0]
    embed_var = feats[2]
    lex_div = feats[5]

    prob = predict_fn(text)
    stability = stability_score(predict_fn, text, perturb_text)

    explanation = []

    # ---------------- PRIMARY SIGNAL ----------------
    if prob > 0.85:
        explanation.append(f"High model probability ({round(prob,3)}) → AI leaning")
        score = 2
    elif prob < 0.25:
        explanation.append(f"Low model probability ({round(prob,3)}) → Human leaning")
        score = -2
    else:
        explanation.append(f"Moderate probability ({round(prob,3)}) → Uncertain")
        score = 0

    # ---------------- SECONDARY SIGNALS ----------------

    #  Stability (based on YOUR data)
    if stability > 0.005:
        explanation.append("High instability → model sensitive → AI signal")
        score += 0.75
    elif stability < 0.0005:
        explanation.append("Very stable → consistent prediction → human signal")
        score -= 0.5
    else:
        explanation.append("Moderate stability → neutral")

    # Perplexity
    if perplexity < 30:
        explanation.append("Very low perplexity → highly predictable (AI signal)")
        score += 1
    elif perplexity > 80:
        explanation.append("High perplexity → irregular/creative (human signal)")
        score -= 1
    else:
        explanation.append("Moderate perplexity → neutral")

    #  Embedding variation (UPDATED RANGE)
    if embed_var < 0.55:
        explanation.append("Low semantic variation → structured text (AI signal)")
        score += 1
    elif embed_var > 0.65:
        explanation.append("High semantic variation → diverse expression (human signal)")
        score -= 1
    else:
        explanation.append("Moderate semantic variation → neutral")

    # Lexical diversity
    if lex_div > 0.6:
        explanation.append("High lexical diversity → human-like richness")
        score -= 1
    elif lex_div < 0.4:
        explanation.append("Low lexical diversity → repetition (AI signal)")
        score += 1
    else:
        explanation.append("Moderate lexical diversity → neutral")

    # ---------------- FINAL DECISION ----------------
    if score >= 1.4:
        decision = "AI Generated"
    elif score <= -1.5:
        decision = "Human Written"
    else:
        decision = "Uncertain"

    return {
        "decision": decision,
        "probability": round(prob, 3),
        "stability": round(stability, 6),
        "score": round(score, 2),
        "explanation": explanation
    }