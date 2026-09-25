#!/usr/bin/env python3
"""Run fixed-input repeated-generation and model-sensitivity experiments."""
from __future__ import annotations
import argparse,json,shlex,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import canonical_hash,write_json
from compile_program import compile_family
from induction.backends import CommandBackend,LocalBackend
from induction.induce import execute

def backend_from_config(row):
    kind=row["backend"]
    if kind=="command":
        command=str(row.get("command","")).strip()
        if not command:
            raise ValueError(f'command is blank for {row.get("name","unnamed model")}; provide an external API wrapper')
        return CommandBackend(shlex.split(command),float(row.get("timeout",300)))
    if kind=="local": return LocalBackend(row["model_path"],int(row.get("max_new_tokens",2048)))
    raise ValueError(f"unsupported backend: {kind}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--request",type=Path,required=True)
    p.add_argument("--models",type=Path,required=True); p.add_argument("--repetitions",type=int,default=50)
    p.add_argument("--output-dir",type=Path,required=True); args=p.parse_args()
    request=json.loads(args.request.read_text()); configs=json.loads(args.models.read_text())["models"]
    args.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for model in configs:
        backend=backend_from_config(model)
        for generation in range(args.repetitions):
            archive=execute(request,backend,model["name"],model.get("revision"),allow_invalid=True)
            if archive["final_executable"]:
                _,programs,_=compile_family(archive["structured_output"])
                archive["compiled_family_sha256"]=canonical_hash([p.candidate_id for p in programs])
            else:
                archive["compiled_family_sha256"]=None
            archive_path=args.output_dir/f'{model["name"]}_{generation:03d}.json'
            write_json(archive_path,archive)
            rows.append({"model":model["name"],"generation_index":generation,
                         "first_pass_valid":archive["first_pass_valid"],"repair_applied":archive["repair_applied"],
                         "final_executable":archive["final_executable"],"latency_seconds":archive["latency_seconds"],
                         "input_tokens":archive["usage"]["input_tokens"],"output_tokens":archive["usage"]["output_tokens"],
                         "response_sha256":archive["response_sha256"],
                         "structured_output_sha256":archive["structured_output_sha256"],
                         "compiled_family_sha256":archive["compiled_family_sha256"]})
    write_json(args.output_dir/"generation_records.json",{"request_sha256":canonical_hash(request),
               "repetitions_per_model":args.repetitions,"records":rows})
if __name__=="__main__": main()
