import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
encoder = joblib.load("onehot_encoder.joblib")
mapping = {
    "nofraud": 0,
    "Phishing": 1,
    "Account Takeover": 2,
    "Synthetic Identity": 3,
    "Card Cloning": 4,
    "Friendly Fraud": 5,
    'Identity Theft': 6
}
cat_cols = [
    'city',
    'merchant_category',
    'country',
    'payment_method',
    'device_type']
def train_encoder(df, cat_cols, path="onehot_encoder.joblib"):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df[cat_cols])
    joblib.dump(encoder, path)
    return encoder
if __name__ == "__main__":
    df = pd.read_csv("bank_fraud.csv")
    train_encoder(df, cat_cols)
def clean(df):
    df=df.drop(columns=['transaction_id'], errors="ignore")
    df=df.drop(columns=['customer_id'], errors="ignore")
    df=df.drop(columns=['is_fraud'], errors="ignore")
    df=df.drop(columns=['hour_of_day'], errors="ignore")
    df=df.drop(columns=['is_weekend'], errors="ignore")
    df=df.drop(columns=['is_night_transaction'], errors="ignore")
    df["transaction_datetime"] = pd.to_datetime(df["transaction_date"] + " " + df["transaction_time"])
    df["year"] = df["transaction_datetime"].dt.year
    df["month"] = df["transaction_datetime"].dt.month
    df["day_of_week"] = df["transaction_datetime"].dt.dayofweek
    df["hour"] = df["transaction_datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df=df.drop(columns=['hour'], errors="ignore")
    df=df.drop(columns=['transaction_date'], errors="ignore")
    df=df.drop(columns=['transaction_time'], errors="ignore")
    df=df.drop(columns=['transaction_datetime'], errors="ignore")
    encoded = encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df
def encode(df):
    X=df.drop(columns=['fraud_type'], errors="ignore")
    y=df.fraud_type.fillna('nofraud').map(mapping)
    X=X.astype('float32')
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    return X_tensor,y_tensor
def clean1(df):
    df["transaction_datetime"] = pd.to_datetime(
        df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str)
    )
    df["year"] = df["transaction_datetime"].dt.year
    df["month"] = df["transaction_datetime"].dt.month
    df["day_of_week"] = df["transaction_datetime"].dt.dayofweek
    df["hour"] = df["transaction_datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.drop(columns=['hour'], errors="ignore")
    df = df.drop(columns=['transaction_date'], errors="ignore")
    df = df.drop(columns=['transaction_time'], errors="ignore")
    df = df.drop(columns=['transaction_datetime'], errors="ignore")
    encoded = encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df
def encode1(df):
    X = df.drop(columns=['fraud_type'], errors="ignore")
    X = X.astype('float32')
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    return X_tensor