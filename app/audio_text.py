"""Explainable lexical examples only, not symptoms, entities or clinical findings."""

import re
import unicodedata

TERMS = {
    "dor no peito": ("dor no peito", "dor toracica", "chest pain"),
    "falta de ar": ("falta de ar", "dispneia", "shortness of breath"),
    "palpitação": ("palpitacao", "palpitacoes", "palpitation", "palpitations"),
    "tontura": ("tontura", "tonturas", "dizziness"),
    "pressão arterial": ("pressao arterial", "blood pressure"),
    "frequência cardíaca": ("frequencia cardiaca", "heart rate"),
}
NEGATION = re.compile(r"\b(nao|sem|nega|nego|negou|ausencia de|no|not|without|denies|deny)\b")
UNCERTAINTY = re.compile(r"\b(talvez|possivel|suspeita|pode|poderia|se|maybe|possible|if|might)\b")
RESET = re.compile(r"[.!?;,:\n]|\b(?:mas|porem|contudo|but|however)\b")
POSITIVE = {"bem", "melhor", "calmo", "calma", "tranquilo", "tranquila", "good", "better", "calm"}
NEGATIVE = {"mal", "pior", "preocupado", "preocupada", "medo", "bad", "worse", "worried", "afraid"}


def normalize(text):
    # NFC first keeps an accented character aligned to one position in the evidence.
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))


def term_rows(segments):
    """Bounded negation scope: last five words in the same clause/ASR segment.

    The label intentionally says 'pista' instead of asserting symptom absence.
    Negation and segmentation are fallible; the full source segment is shown.
    """
    rows = []
    for segment in segments:
        original = unicodedata.normalize("NFC", segment.text)
        text = normalize(original)
        for label, aliases in TERMS.items():
            for alias in aliases:
                for match in re.finditer(r"\b" + re.escape(alias) + r"\b", text):
                    prefix = RESET.split(text[:match.start()])[-1]
                    context = " ".join(prefix.split()[-5:])
                    # 'não sei se' is uncertainty, not denial of a symptom.
                    if UNCERTAINTY.search(context):
                        state = "Menção com pista de incerteza"
                    elif NEGATION.search(context):
                        state = "Menção com pista de negação"
                    else:
                        state = "Menção sem pista de negação"
                    rows.append({
                        "Início (s)": segment.start, "Fim (s)": segment.end,
                        "Termo do exemplo": label,
                        "Expressão encontrada": original[match.start():match.end()],
                        "Leitura lexical": state,
                        "Trecho original": original,
                    })
    return rows


def sentiment_summary(text):
    """Count lexical words, never infer emotion from voice or issue a diagnosis.

    Terms preceded by negation/uncertainty in the same small scope are excluded,
    not reversed. No matched words does NOT mean neutral sentiment.
    """
    normalized = normalize(unicodedata.normalize("NFC", text))
    positive, negative, excluded = [], [], []
    for match in re.finditer(r"\b\w+\b", normalized):
        word = match.group()
        if word not in POSITIVE | NEGATIVE:
            continue
        prefix = RESET.split(normalized[:match.start()])[-1]
        context = " ".join(prefix.split()[-3:])
        if NEGATION.search(context) or UNCERTAINTY.search(context):
            excluded.append(word)
        elif word in POSITIVE:
            positive.append(word)
        else:
            negative.append(word)
    total = len(positive) + len(negative)
    score = (len(positive) - len(negative)) / total if total else None
    return {"positive": positive, "negative": negative, "excluded": excluded, "score": score}
