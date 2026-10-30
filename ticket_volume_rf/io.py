"""Load a monthly panel from a CSV with the public column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Optional headers accepted in addition to the canonical names. No
# organisation-specific columns are dropped or required.
COLUMN_ALIASES = {
    "Ticket Volume": "ticket_volume",
    "Sales Market+": "region",
    "iter": "series_id",
}


def load_panel(path: str | Path) -> pd.DataFrame:
    """Read a monthly panel and rename known aliases."""
    frame = pd.read_csv(path)
    rename = {source: target for source, target in COLUMN_ALIASES.items() if source in frame.columns}
    if rename:
        frame = frame.rename(columns=rename)
    return frame
