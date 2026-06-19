"""Final round of probes for a broader constraint shadow-price report.

If none of these return real shadow prices distinct from PRC_RTM_NOMOGRAM,
we conclude the nomogram+intertie panel is comprehensive.
"""
import io
import time
import zipfile

import pandas as pd
import requests

OASIS_URL = "http://oasis.caiso.com/oasisapi/SingleZip"
START = "20260615T07:00-0000"
END = "20260616T07:00-0000"

CANDIDATES = [
    ("PRC_BC", 1, {"market_run_id": "RTM"}),
    ("PRC_BC", 6, {"market_run_id": "RTM"}),
    ("PRC_BINDING_CONSTRAINT", 1, {"market_run_id": "RTM"}),
    ("PRC_CMP", 1, {"market_run_id": "RTM"}),
    ("PRC_RTM_BC", 1, None),
    ("PRC_RTM_NOMOGRAM", 1, None),
]


def try_endpoint(queryname, version, extra):
    params = {
        "resultformat": 6,
        "queryname": queryname,
        "version": version,
        "startdatetime": START,
        "enddatetime": END,
    }
    if extra:
        params.update(extra)
    print(f"\n-- {queryname} v{version} {extra or ''} --")
    try:
        resp = requests.get(OASIS_URL, params=params, timeout=60)
    except Exception as e:
        print(f"  HTTP error: {type(e).__name__}: {e}")
        return
    print(f"  status {resp.status_code}, {len(resp.content)} bytes")
    if resp.status_code != 200 or not resp.content.startswith(b"PK"):
        print(f"  head: {resp.content[:160]!r}")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            print(f"  zip contains: {names}")
            for n in names:
                if "INVALID" in n.upper():
                    with zf.open(n) as f:
                        head = f.read(200)
                    print(f"  invalid: {head[:200]!r}")
                    continue
                if n.endswith(".csv"):
                    with zf.open(n) as f:
                        df = pd.read_csv(f)
                    print(f"  parsed CSV: {len(df)} rows")
                    print(f"  columns: {list(df.columns)}")
                    if not df.empty:
                        print(df.head(2).to_string())
    except Exception as e:
        print(f"  zip/parse error: {type(e).__name__}: {e}")


for q, v, extra in CANDIDATES:
    try_endpoint(q, v, extra)
    time.sleep(5)
