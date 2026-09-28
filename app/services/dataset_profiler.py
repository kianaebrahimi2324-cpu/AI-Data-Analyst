import pandas as pd


def build_dataset_profile(df: pd.DataFrame) -> dict:
    """Create a deterministic profile of the dataset."""

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    missing_values = (
        df.isnull()
        .sum()
        .to_dict()
    )

    missing_percent = (
        df.isnull()
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),

        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,

        "missing_values": missing_values,
        "missing_percent": missing_percent,

        "duplicate_rows": int(
            df.duplicated().sum()
        ),
    }

    return profile