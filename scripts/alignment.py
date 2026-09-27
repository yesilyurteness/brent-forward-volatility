"""GPR alignment switch shared by the model and analysis scripts.

--gpr-alignment timestamp   (default) reads outputs/features.csv and writes the original,
                            unsuffixed output names. GPR features use the observation of
                            the previous trading day (timestamp alignment; paper
                            Appendix A).
--gpr-alignment publication reads outputs/features_publication_aligned.csv and appends
                            "_publication_aligned" to every output AND to every upstream
                            output it reads, so the two versions never mix inside a chain
                            (PRIMARY results; see 02_build_features._align_gpr).

Only the GPR-derived feature columns differ between the two feature files; every other
column, every row and the first fully valid row (127) are identical, so both versions are
evaluated on exactly the same samples. check_equal() turns that into assertions in
publication mode:
  * same sample : the row keys (horizon, fold, dates, train/test sizes) match the
                  timestamp output exactly;
  * unaffected  : outputs of models that use no GPR input match it bit for bit.
"""
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

ALIGNMENTS = ("timestamp", "publication")
_SUFFIX = {"timestamp": "", "publication": "_publication_aligned"}
_FEATURES = {"timestamp": "features.csv",
             "publication": "features_publication_aligned.csv"}


def add_argument(ap):
    ap.add_argument("--gpr-alignment", choices=ALIGNMENTS, default="timestamp",
                    help="timestamp: original GPR alignment, unsuffixed outputs. "
                         "publication: GPR aligned to its publication dates, outputs "
                         "and upstream inputs carry the _publication_aligned suffix.")


def suffix(alignment):
    return _SUFFIX[alignment]


def features_path(alignment):
    return OUT_DIR / _FEATURES[alignment]


def out(name, alignment):
    """outputs/<stem><suffix><ext> for a base file name such as 'wf_metrics_all.csv'."""
    p = Path(name)
    return OUT_DIR / f"{p.stem}{_SUFFIX[alignment]}{p.suffix}"


def check_equal(new, base_name, keys, cols, alignment, rows=None, what=""):
    """Asserts that `new` equals the unsuffixed timestamp file on keys + cols, bit for bit.

    rows : optional function frame -> boolean mask, applied to BOTH frames (e.g. select
           the GPR-free models, or the horizons/folds covered by a smoke run).
    No-op in timestamp mode. Returns the number of rows compared.
    """
    if alignment == "timestamp":
        return 0
    base = pd.read_csv(OUT_DIR / base_name, float_precision="round_trip")
    # Pass the new frame through the same CSV round trip as the stored file, so both
    # sides are parsed identically (dtypes, dates as text, float repr).
    new = pd.read_csv(StringIO(new.to_csv(index=False)), float_precision="round_trip")
    if rows is not None:
        base = base[rows(base)]
        new = new[rows(new)]
    use = list(keys) + [c for c in cols if c not in keys]
    a = base[use].sort_values(list(keys)).reset_index(drop=True)
    b = new[use].sort_values(list(keys)).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0, (
        f"{base_name} [{what}]: row counts differ ({len(a)} vs {len(b)})")
    for c in use:
        if a[c].dtype != b[c].dtype:
            b[c] = b[c].astype(a[c].dtype)
    pd.testing.assert_frame_equal(a, b, check_exact=True,
                                  obj=f"{base_name} [{what}]")
    print(f"[kontrol] {base_name} [{what}]: {len(a)} satir, timestamp surumuyle "
          "BIT DUZEYINDE AYNI")
    return len(a)
