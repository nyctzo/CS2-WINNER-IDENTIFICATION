import pandas as pd

def load_and_prepare_data(path):
    # Load ARFF-like dataset
    data = []
    columns = []

    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line.startswith("@ATTRIBUTE"):
                columns.append(line.split(" ")[1])
            elif not (line.startswith("@") or line.startswith("%") or line == ""):
                data.append(line.split(","))

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop non-numeric columns
    df_numeric = df.drop(columns=["map", "round_winner"], errors="ignore")

    # Correlation
    correlations = df_numeric.corr()

    # Feature selection
    selected_columns = []
    for col in df_numeric.columns:
        if col != "t_win" and abs(correlations[col]["t_win"]) > 0.15:
            selected_columns.append(col)

    X = df[selected_columns]
    y = df["t_win"]

    return X, y


