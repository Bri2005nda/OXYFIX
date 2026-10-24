# -*- coding: utf-8 -*-
"""Télécharge le binaire Windows officiel de MMseqs2 (dernière release GitHub)."""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
API = "https://api.github.com/repos/soedinglab/MMseqs2/releases/latest"


def main():
    TOOLS.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(API, headers={"User-Agent": "oxyfix"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        rel = json.load(fh)
    print("release :", rel.get("tag_name"))
    asset = None
    for a in rel.get("assets", []):
        name = a["name"].lower()
        if "win" in name and name.endswith(".zip"):
            asset = a
            break
    if asset is None:
        print("AUCUN asset Windows dans la release ; assets :", [a["name"] for a in rel.get("assets", [])])
        sys.exit(1)
    dest = TOOLS / asset["name"]
    print("téléchargement :", asset["name"], f"({asset['size'] / 1e6:.0f} Mo)")
    if not dest.exists() or dest.stat().st_size != asset["size"]:
        urllib.request.urlretrieve(asset["browser_download_url"], dest)
    print("extraction ...")
    with zipfile.ZipFile(dest) as zf:
        zf.extractall(TOOLS)
    exes = list(TOOLS.rglob("mmseqs.exe"))
    print("binaire :", exes[0] if exes else "INTROUVABLE")


if __name__ == "__main__":
    main()
