import numpy as np
import pandas as pd
import pytest
from io import StringIO
from unittest.mock import patch


RAW_COLUMNS = {
    "AI 1/AI 1 (m/s2)": "Acc1 m/s2",
    "AI 2/AI 2 (m/s2)": "Acc2 m/s2",
    "AI 3/AI 3 (V)": "Força (V)",
    "AI 4/AI 4 (V)": "Disp (V)",
}

RENAME_MAP = RAW_COLUMNS


def make_sample_df(times=None):
    """Create a DataFrame that mimics the structure produced after read_csv + set_index + rename."""
    if times is None:
        times = np.linspace(0.0, 5.0, 500)
    data = {
        "Acc1 m/s2": np.sin(2 * np.pi * 5 * times),
        "Acc2 m/s2": np.cos(2 * np.pi * 5 * times),
        "Força (V)": np.ones_like(times) * 0.5,
        "Disp (V)": np.ones_like(times) * 1.2,
    }
    df = pd.DataFrame(data, index=pd.Index(times, name="Time (s)"))
    return df


# ---------------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------------

class TestColumnRenaming:
    def _raw_df(self):
        times = np.linspace(0.0, 5.0, 50)
        data = {col: np.zeros(50) for col in RAW_COLUMNS}
        df = pd.DataFrame(data, index=pd.Index(times, name="Time (s)"))
        return df

    def test_rename_produces_expected_columns(self):
        df = self._raw_df()
        df.rename(columns=RENAME_MAP, inplace=True)
        assert set(df.columns) == set(RENAME_MAP.values())

    def test_original_column_names_removed_after_rename(self):
        df = self._raw_df()
        df.rename(columns=RENAME_MAP, inplace=True)
        for old_col in RENAME_MAP:
            assert old_col not in df.columns

    def test_index_name_preserved_after_rename(self):
        df = self._raw_df()
        df.rename(columns=RENAME_MAP, inplace=True)
        assert df.index.name == "Time (s)"


# ---------------------------------------------------------------------------
# Data slicing (region of interest 3.1 – 4.25 s)
# ---------------------------------------------------------------------------

class TestDataSlicing:
    T_START = 3.1
    T_END = 4.25

    def test_slice_returns_correct_time_range(self):
        df = make_sample_df()
        sliced = df.loc[self.T_START: self.T_END]
        assert sliced.index.min() >= self.T_START
        assert sliced.index.max() <= self.T_END

    def test_slice_is_not_empty(self):
        df = make_sample_df()
        sliced = df.loc[self.T_START: self.T_END]
        assert len(sliced) > 0

    def test_u_np_shape_matches_time_np(self):
        df = make_sample_df()
        u_np = np.array(df.loc[self.T_START: self.T_END]["Acc1 m/s2"])
        time_np = np.array(df.loc[self.T_START: self.T_END].index)
        assert u_np.shape == time_np.shape

    def test_u_np_values_match_slice(self):
        df = make_sample_df()
        sliced = df.loc[self.T_START: self.T_END]
        u_np = np.array(sliced["Acc1 m/s2"])
        np.testing.assert_array_equal(u_np, sliced["Acc1 m/s2"].values)

    def test_time_np_is_monotonically_increasing(self):
        df = make_sample_df()
        time_np = np.array(df.loc[self.T_START: self.T_END].index)
        assert np.all(np.diff(time_np) > 0)

    def test_slice_outside_range_is_empty(self):
        df = make_sample_df(times=np.linspace(0.0, 2.0, 200))
        sliced = df.loc[self.T_START: self.T_END]
        assert len(sliced) == 0


# ---------------------------------------------------------------------------
# CSV loading (mocked)
# ---------------------------------------------------------------------------

class TestCsvLoading:
    CSV_CONTENT = (
        "\n" * 10  # skiprows=10
        + "Time (s)\tAI 1/AI 1 (m/s2)\tAI 2/AI 2 (m/s2)\tAI 3/AI 3 (V)\tAI 4/AI 4 (V)\n"
        + "0.0\t1.0\t2.0\t3.0\t4.0\n"
        + "0.1\t1.1\t2.1\t3.1\t4.1\n"
    )

    def _load(self, content):
        lines = content.split("\n")
        data_lines = "\n".join(lines[10:])
        df = pd.read_csv(StringIO(data_lines), sep="\t")
        df.set_index("Time (s)", inplace=True)
        df.rename(columns=RENAME_MAP, inplace=True)
        return df

    def test_loaded_df_has_renamed_columns(self):
        df = self._load(self.CSV_CONTENT)
        assert "Acc1 m/s2" in df.columns
        assert "Acc2 m/s2" in df.columns

    def test_loaded_df_index_name(self):
        df = self._load(self.CSV_CONTENT)
        assert df.index.name == "Time (s)"

    def test_loaded_df_row_count(self):
        df = self._load(self.CSV_CONTENT)
        assert len(df) == 2

    def test_loaded_acc1_values(self):
        df = self._load(self.CSV_CONTENT)
        np.testing.assert_array_almost_equal(df["Acc1 m/s2"].values, [1.0, 1.1])
