#!/usr/bin/env python3
"""Deterministic JSON extraction; it never invents missing semantic fields."""
from __future__ import annotations
import json,re

def parse_or_repair(text: str):
    try: return json.loads(text),False
    except json.JSONDecodeError: pass
    fenced=re.search(r"\x60{3}(?:json)?\s*(\{.*\})\s*\x60{3}",text,re.S)
    if fenced:
        return json.loads(fenced.group(1)),True
    start=text.find("{"); end=text.rfind("}")
    if start>=0 and end>start:
        return json.loads(text[start:end+1]),True
    raise ValueError("no JSON object found")
