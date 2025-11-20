import os
import numpy as np
import pandas as pd
from src.model.matrix_factorization import MatrixFactorization
import joblib
import argparse

def load_sample_data(path="data/processed/sample_data.csv"):
    # sample CSV with columns: user_id,item_id,rating
    df = pd.read_csv(path)
    return df

def build_interactions(df):
    users = df['user_id'].unique().tolist()
    items = df['item_id'].unique().tolist()
    user_map = {u:i for i,u in enumerate(users)}
    item_map = {v:i for i,v in enumerate(items)}
    interactions = []
    for _, row in df.iterrows():
        interactions.append((user_map[row['user_id']], item_map[row['item_id']], row['rating']))
    return interactions, len(users), len(items), user_map, item_map

def main(args):
    df = load_sample_data()
    interactions, n_users, n_items, user_map, item_map = build_interactions(df)
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(interactions)}")
    mf = MatrixFactorization(n_users, n_items, n_factors=16, lr=args.lr, reg=args.reg)
    loss = mf.train(interactions, epochs=args.epochs, batch_size=args.batch)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({'U':mf.U,'V':mf.V,'user_map':user_map,'item_map':item_map}, "artifacts/model.joblib")
    print("Model artifacts saved to artifacts/model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
