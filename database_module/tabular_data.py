from pathlib import Path

import pandas as pd


SUPPORTED_DATASET_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def read_tabular_dataset(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path)
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    supported = ", ".join(sorted(SUPPORTED_DATASET_EXTENSIONS))
    raise ValueError(f"Unsupported dataset file type. Use one of: {supported}")
