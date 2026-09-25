#!/usr/bin/env python3
"""Summarize generation validity, executability, latency, and token usage."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--records",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    rows=json.loads(args.records.read_text())["records"]; groups=defaultdict(list)
    for row in rows: groups[row["model"]].append(row)
    summary={}
    for model,items in groups.items():
        latency=np.asarray([row["latency_seconds"] for row in items],float)
        input_tokens=[row["input_tokens"] for row in items if row["input_tokens"] is not None]
        output_tokens=[row["output_tokens"] for row in items if row["output_tokens"] is not None]
        summary[model]={"n":len(items),"first_pass_valid_percent":100*np.mean([row["first_pass_valid"] for row in items]),
                        "final_executable_percent":100*np.mean([row["final_executable"] for row in items]),
                        "repair_count":sum(row["repair_applied"] for row in items),
                        "latency_seconds_mean":float(latency.mean()),"latency_seconds_min":float(latency.min()),
                        "latency_seconds_max":float(latency.max()),
                        "input_tokens_mean":None if not input_tokens else float(np.mean(input_tokens)),
                        "output_tokens_mean":None if not output_tokens else float(np.mean(output_tokens))}
    write_json(args.output,{"models":summary})
if __name__=="__main__": main()
