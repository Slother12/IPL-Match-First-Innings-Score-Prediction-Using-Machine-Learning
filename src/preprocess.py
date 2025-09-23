# src/preprocess.py
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame"""
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: drop duplicates and fix column names"""
    df = df.drop_duplicates().reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]  # remove spaces
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for IPL dataset.
    This is just an example — you will adapt it depending on your dataset columns.
    """

    # Example: convert teams, venues into dummy variables
    if 'batting_team' in df.columns and 'bowling_team' in df.columns:
        df = pd.get_dummies(df, columns=['batting_team', 'bowling_team'], drop_first=True)

    if 'venue' in df.columns:
        df = pd.get_dummies(df, columns=['venue'], drop_first=True)

    return df


if __name__ == "__main__":
    # Example usage: run python src/preprocess.py
    df = load_csv("data/raw/matches.csv")   # adjust dataset path
    df = basic_clean(df)
    df = feature_engineer(df)
    df.to_csv("data/processed/processed_matches.csv", index=False)
    print("✅ Data preprocessing complete. Saved to data/processed/processed_matches.csv")
