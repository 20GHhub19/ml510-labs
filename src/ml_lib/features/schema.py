"""Name the numeric and categorical inputs used by a tabular model."""
from dataclasses import dataclass
import pandas as pd
from ml_lib.data.validation import require_columns


@dataclass(frozen=True)
class FeatureSchema:
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return list(self.numeric + self.categorical)

    def validate(self, X: pd.DataFrame) -> None:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Feature names must be nonempty and unique across schema groups.")
        require_columns(X, self.columns)
