import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_data():
    df = pd.read_csv("data/cleaned_accident_data.csv")

    # Convert date into useful numerical features
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Remove original date
    df = df.drop(columns=["date"])

    # Remove time because hour already exists
    df = df.drop(columns=["time"])

    # Remove post-accident information
    df = df.drop(columns=["casualties", "vehicles_involved"])

    # Create risk level from risk score
    def get_risk_level(score):
        if score <= 0.33:
            return "Low"
        elif score <= 0.66:
            return "Medium"
        else:
            return "High"

    df["risk_level"] = df["risk_score"].apply(get_risk_level)

    # Separate features and target
    X = df.drop(columns=["accident_severity", "risk_score", "risk_level", "cause"])
    y = df["risk_level"]

    return X, y


def create_preprocessor(X):

    categorical_columns = [
    "city",
    "state",
    "day_of_week",
    "road_type",
    "weather",
    "visibility",
    "traffic_density"
]

    numerical_columns = [
        column for column in X.columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    return preprocessor
if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_accident_data.csv")

    print("Cleaned Dataset:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)