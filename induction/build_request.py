#!/usr/bin/env python3
"""Build the fixed structured induction request from task, specification, and diagnostics."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import write_json
from induction.schema import output_schema,validate_request

def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",type=Path,required=True)
    p.add_argument("--specification",type=Path,required=True); p.add_argument("--diagnostics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    payload={"task_description":args.task.read_text().strip(),
             "evidence_specification":json.loads(args.specification.read_text()),
             "training_diagnostics":json.loads(args.diagnostics.read_text()),"output_schema":output_schema()}
    validate_request(payload); write_json(args.output,payload)
if __name__=="__main__": main()
