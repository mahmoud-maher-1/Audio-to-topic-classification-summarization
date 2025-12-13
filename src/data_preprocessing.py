import pandas as pd
import numpy as np
import re
import string
import random
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pycountry

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Cleans and preprocesses raw text by lowercasing, removing noise,
    tokenizing, removing stopwords, and lemmatizing.
    """

    def to_lower(t):
        return t.lower()

    def remove_punctuation_numbers(t):
        t = re.sub(r'\d+', '', t)
        t = re.sub(r'\b(\d+)(st|nd|rd|th)\b', '', t)
        t = re.sub(
            r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
            '', t)
        return t.translate(str.maketrans('', '', string.punctuation))

    def tokenize(t):
        return nltk.word_tokenize(t)

    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stop_words]

    def apply_lemmatization(tokens):
        return [lemmatizer.lemmatize(word) for word in tokens]

    text = to_lower(text)
    text = remove_punctuation_numbers(text)

    # Remove country names to avoid bias
    for c in pycountry.countries:
        if c.name.lower() in text:
            text = re.sub(r'\b' + re.escape(c.name.lower()) + r'\b', '', text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_lemmatization(tokens)
    return " ".join(tokens)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lem in syn.lemmas():
            if lem.name().lower() != word.lower():
                synonyms.add(lem.name().replace("_", " "))
    return list(synonyms)


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    if len(words) < 2: return sentence
    candidates = [w for w in words if get_synonyms(w)]
    if not candidates: return sentence
    for _ in range(n):
        word = random.choice(candidates)
        synonym = random.choice(get_synonyms(word))
        words = [synonym if w == word else w for w in words]
    return " ".join(words)


def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1: return sentence
    new_words = [w for w in words if random.random() > p]
    return " ".join(new_words) if new_words else random.choice(words)


def random_swap(sentence, n=1):
    words = sentence.split()
    if len(words) < 2: return sentence
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def augment_text(text):
    """Randomly applies augmentation techniques."""
    choice = random.choice(["synonym", "swap", "delete"])
    if choice == "synonym":
        return synonym_replacement(text, n=1)
    elif choice == "swap":
        return random_swap(text, n=1)
    elif choice == "delete":
        return random_deletion(text, p=0.15)
    return text


def load_and_preprocess_data(filepath):
    """Loads dataset, handles augmentation for class balancing, and preprocesses text."""
    df = pd.read_csv(filepath)

    # Create generalized labels
    type_mapping = {
        "Hygiene": "Health", "Health Monitoring": "Health", "Health Testing": "Health",
        "COVID-19 Vaccines": "Health", "Health Resources": "Health",
        "Closure and Regulation of Schools": "Education", "Quarantine": "Restrictions",
        "Social Distancing": "Restrictions", "Lockdown": "Restrictions", "Curfew": "Restrictions",
        "Internal Border Restrictions": "Restrictions", 'External Border Restrictions': "Restrictions",
        'Restriction and Regulation of Businesses': "Restrictions",
        "Restriction and Regulation of Government Services": "Restrictions",
        "Restrictions of Mass Gatherings": "Restrictions", "Declaration of Emergency": "Government Measures",
        "New Task Force, Bureau or Administrative Configuration": "Government Measures",
        "Public Awareness Measures": "Awareness", "Anti-Disinformation Measures": "Awareness",
        "Other Policy Not Listed Above": "Other"
    }
    df['type_gen'] = df['type'].map(type_mapping)
    df = df.drop_duplicates(subset=["description"])

    # Augment data to balance classes
    class_counts = df["type"].value_counts()
    max_count = class_counts.max()
    augmented_rows = []

    for label, count in class_counts.items():
        subset = df[df["type"] == label]
        needed = max_count - count
        if needed > 0:
            for _ in range(needed):
                row = subset.sample(1).iloc[0]
                new_desc = augment_text(row["description"])
                augmented_rows.append({
                    "description": new_desc,
                    "type": label,
                    "type_gen": type_mapping.get(label, "Other")
                })

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, aug_df], ignore_index=True)

    # Apply text cleaning
    df["description"] = df["description"].apply(preprocess_text)

    return df

def load_and_augment_data_for_visualization(filepath):
    """Loads dataset, handles augmentation for class balancing, and sends the augmented dataset for visualization."""
    df = pd.read_csv(filepath)

    df = df.drop_duplicates(subset=["description"])

    # Augment data to balance classes
    class_counts = df["type"].value_counts()
    max_count = class_counts.max()
    augmented_rows = []

    for label, count in class_counts.items():
        subset = df[df["type"] == label]
        needed = max_count - count
        if needed > 0:
            for _ in range(needed):
                row = subset.sample(1).iloc[0]
                new_desc = augment_text(row["description"])
                augmented_rows.append({
                    "description": new_desc,
                    "type": label,
                })

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, aug_df], ignore_index=True)

    return df

def sentence_embed(text_list, model):
    """Generates embeddings using SentenceTransformer."""
    return np.array(model.encode(text_list, show_progress_bar=True))