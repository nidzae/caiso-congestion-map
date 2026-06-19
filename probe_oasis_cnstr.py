"""Probe OASIS SingleZip for the general transmission-constraint report.

We don't know the exact queryname/version, so try a few candidates.
Print outcome for each so we can identify the working endpoint.
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
    ("PRC_INTVL_CNSTR_PRC", 1, {"market_run_id": "RTM"}),
    ("PRC_INTVL_CNSTR_PRC", 6, {"market_run_id": "RTM"}),
    ("PRC_RTM_CNSTR_PRC", 1, None),
    ("PRC_RTM_CNSTR", 1, None),
    ("PRC_CNSTR", 1, {"market_run_id": "RTM"}),
    ("PRC_CNSTR", 6, {"market_run_id": "RTM"}),
    ("PRC_INTVL_CNSTR_PRC", 5, {"market_run_id": "RTM"}),
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
    print(f"  status {resp.status_code}, {len(resp.content)} bytes, content-type={resp.headers.get('content-type')}")
    if resp.status_code != 200:
        print(f"  body head: {resp.text[:200]}")
        return
    # Sometimes OASIS returns an XML error inside a 200 instead of a zip
    if not resp.content.startswith(b"PK"):
        print(f"  not a zip; head: {resp.content[:200]!r}")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            print(f"  zip contains: {names}")
            for n in names:
                with zf.open(n) as f:
                    head = f.read(400)
                print(f"  {n} head: {head[:300]!r}")
                # Try parsing as CSV if it looks like CSV
                if n.endswith(".csv"):
                    with zf.open(n) as f:
                        df = pd.read_csv(f)
                    print(f"  parsed CSV: {len(df)} rows, columns={list(df.columns)}")
    except Exception as e:
        print(f"  zip/parse error: {type(e).__name__}: {e}")


for q, v, extra in CANDIDATES:
    try_endpoint(q, v, extra)
    time.sleep(5)  # OASIS throttling
