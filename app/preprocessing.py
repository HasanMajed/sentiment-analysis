import re
import string
import unicodedata
import emoji
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

negators = {"not", "no", "never", "n't", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt"}
stop_words = set(stopwords.words('english')) - negators
lemmatizer = WordNetLemmatizer()

EMOJI_SENTIMENT = {
    "😊": "positive_emoji", "😄": "positive_emoji", "😁": "positive_emoji", "😍": "positive_emoji",
    "👍": "positive_emoji", "🎉": "positive_emoji", "😎": "positive_emoji",
    "😢": "negative_emoji", "😠": "negative_emoji", "😡": "negative_emoji",
    "👎": "negative_emoji", "💔": "negative_emoji", "😭": "negative_emoji", "😞": "negative_emoji",
    "😐": "neutral_emoji", "😶": "neutral_emoji", "🤔": "neutral_emoji", "😑": "neutral_emoji"
}

def normalize_apostrophes(text):
    text = re.sub(r"[’‘`´]", "'", text)
    text = re.sub(r"\bdont\b", "don't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoesnt\b", "doesn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdidnt\b", "didn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwasnt\b", "wasn't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshouldnt\b", "shouldn't", text, flags=re.IGNORECASE)
    return text

def normalize_emojis(text):
    if not isinstance(text, str):
        return text
    result = []
    seen = set()
    for char in text:
        if unicodedata.name(char, "").startswith("EMOJI MODIFIER FITZPATRICK"):
            continue
        if char in emoji.EMOJI_DATA:
            if char in EMOJI_SENTIMENT:
                result.append(EMOJI_SENTIMENT[char])
            else:
                base = emoji.demojize(char)
                base = re.sub(r"_medium_skin_tone|_light_skin_tone|_dark_skin_tone|_medium-dark_skin_tone|_medium-light_skin_tone", "", base)
                unified = emoji.emojize(base, language='alias')
                if unified not in seen:
                    result.append(unified)
                    seen.add(unified)
        else:
            result.append(char)
    return ''.join(result)

def detect_language(text):
    return 'ar' if re.search(r'[\u0600-\u06FF]', text) else 'en'

def mark_negation(text):
    tokens = text.split()
    out = []
    negate = False
    for tok in tokens:
        low_tok = tok.lower()
        if low_tok in {"not", "no", "never", "n't", "don't", "doesn't", "didn't"}:
            negate = True
            out.append(tok)
            continue
        if negate:
            out.append(f"not_{tok}")
            negate = False
        else:
            out.append(tok)
    return " ".join(out)

def normalize_arabic(text):
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("ـ", "", text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[؟،؛ـ“”«»…!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_english(text):
    text = normalize_apostrophes(text)
    text = text.lower()
    text = mark_negation(text)
    punct = string.punctuation.replace('_', '')
    text = re.sub(f"[{re.escape(punct)}]", "", text)
    text = re.sub(r'([!?.\-])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    text = ' '.join(lemmatizer.lemmatize(w) for w in tokens)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    try:
        text = str(TextBlob(text).correct())
    except Exception:
        pass
    return text

def clean_text(text: str) -> str:
    text = normalize_apostrophes(text)
    text = normalize_emojis(text)
    if detect_language(text) == 'ar':
        return normalize_arabic(text)
    return normalize_english(text)