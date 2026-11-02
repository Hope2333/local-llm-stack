#!/usr/bin/env python3
"""Build the download manifest for every Bonsai-2 GGUF variant not yet local."""
import json, os, subprocess, sys

REPOS = {"2B": "Ternary-Bonsai-2-2B-gguf", "4B": "Ternary-Bonsai-2-4B-gguf",
         "9B": "Ternary-Bonsai-2-9B-gguf", "27B": "Ternary-Bonsai-2-27B-gguf"}
BASE = "https://modelscope.cn/models/prism-ml/{repo}/resolve/master/{fname}"
ROOT = os.path.expanduser("~/models/bonsai2")
SKIP_ABOVE = 20 * 1024**3

rows = []
for fam, repo in REPOS.items():
    url = f"https://modelscope.cn/api/v1/models/prism-ml/{repo}/repo/files?Revision=master&Recursive=true"
    out = subprocess.run(["curl", "-s", "--max-time", "40", url], capture_output=True).stdout
    files = (json.loads(out).get("Data") or {}).get("Files") or []
    for f in files:
        p = f.get("Path", "")
        if not p.endswith(".gguf"):
            continue
        size = f.get("Size")
        if size > SKIP_ABOVE:
            print(f"skip too large: {p} ({size})", file=sys.stderr)
            continue
        rows.append("\t".join([f"{fam}/{p}", str(size), f.get("Sha256"), BASE.format(repo=repo, fname=p)]))

with open("/tmp/bonsai_manifest.tsv", "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"manifest rows: {len(rows)}  total bytes: {sum(int(r.split(chr(9))[1]) for r in rows)}")
