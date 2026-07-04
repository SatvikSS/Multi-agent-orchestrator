"""Streamlit demo UI. Launch with `surgeon ui` or `streamlit run app.py`."""

from pathlib import Path


def app_path() -> Path:
    """Absolute path to the Streamlit app script."""
    return Path(__file__).parent / "app.py"
