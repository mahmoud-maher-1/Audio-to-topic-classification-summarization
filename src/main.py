import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sentence_transformers import SentenceTransformer

import visualizations
from data_preprocessing import load_and_preprocess_data, sentence_embed
from model_training import (
    build_distributed_model, train_distributed_model,
    build_generalized_model, train_generalized_model
)
from evaluation import plot_training_history, evaluate_model

DATASET_PATH = '../Dataset/dataset_raw.csv'
MODEL_DIR = '../Models/'
VIS_DIR = '../Visualizations/'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(DATASET_PATH)

    # --- Execute Visualizations ---
    print("Creating Data Visualizations...")
    visualizations.main(df, VIS_DIR)

    # Encode labels
    print("Encoding labels...")
    le_dist = LabelEncoder()
    y_dist = le_dist.fit_transform(df['type'])

    le_gen = LabelEncoder()
    y_gen = le_gen.fit_transform(df['type_gen'])

    joblib.dump(le_dist, os.path.join(MODEL_DIR, 'label_encoder_dist.joblib'))
    joblib.dump(le_gen, os.path.join(MODEL_DIR, 'label_encoder_gen.joblib'))

    print("Generating sentence embeddings...")
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = sentence_embed(df['description'].tolist(), sentence_model)

    # Splits
    X_train, X_test, y_train_d, y_test_d = train_test_split(
        embeddings, y_dist, test_size=0.2, random_state=42, stratify=y_dist
    )
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        embeddings, y_gen, test_size=0.2, random_state=42, stratify=y_gen
    )

    y_train_d_cat = to_categorical(y_train_d)
    y_train_g_cat = to_categorical(y_train_g)

    # --- Train Distributed Types Model ---
    print("Training Distributed Types Model...")
    model_dist = build_distributed_model(X_train.shape[1], len(le_dist.classes_))
    # Uses specific callbacks/params (batch=32, epochs=100)
    hist_dist = train_distributed_model(model_dist, X_train, y_train_d_cat)

    model_dist.save(os.path.join(MODEL_DIR, 'distributed_types_20classes.keras'))
    plot_training_history(hist_dist, 'Distributed Types Model', os.path.join(VIS_DIR, 'history_dist.png'))
    evaluate_model(model_dist, X_test, y_test_d, le_dist, 'Distributed Types', os.path.join(VIS_DIR, 'cm_dist.png'))

    # --- Train Generalized Types Model ---
    print("Training Generalized Types Model...")
    model_gen = build_generalized_model(X_train_g.shape[1], len(le_gen.classes_))
    # Uses specific callbacks/params (batch=256, epochs=50)
    hist_gen = train_generalized_model(model_gen, X_train_g, y_train_g_cat)

    model_gen.save(os.path.join(MODEL_DIR, 'generalized_types_6classes.keras'))
    plot_training_history(hist_gen, 'Generalized Types Model', os.path.join(VIS_DIR, 'history_gen.png'))
    evaluate_model(model_gen, X_test_g, y_test_g, le_gen, 'Generalized Types', os.path.join(VIS_DIR, 'cm_gen.png'))

    print("Pipeline complete. Models and visualizations saved.")


if __name__ == "__main__":
    main()