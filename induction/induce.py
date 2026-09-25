#!/usr/bin/env python3
"""Execute structured induction through external-command or local backends."""
from __future__ import annotations
import argparse,json,shlex,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import write_json
from induction.archive import archive_payload
from induction.backends import CommandBackend,LocalBackend
from induction.model_catalog import DEFAULT_MODEL,SUPPORTED_MODELS
from induction.repair import parse_or_repair
from induction.schema import canonical_request_text,validate_output,validate_request

def make_backend(args):
    if args.backend=="command": return CommandBackend(shlex.split(args.command),args.timeout)
    if args.backend=="local": return LocalBackend(args.model_path,args.max_new_tokens)
    raise ValueError(args.backend)

def execute(request,backend,model_name=None,model_revision=None,allow_invalid=False):
    validate_request(request); prompt=canonical_request_text(request); response=backend.generate(prompt)
    first_pass_valid=True
    error=None
    try:
        structured=json.loads(response.text); validate_output(structured); repaired=False
    except (json.JSONDecodeError,ValueError,KeyError,TypeError):
        first_pass_valid=False
        try:
            structured,repaired=parse_or_repair(response.text); validate_output(structured)
        except (json.JSONDecodeError,ValueError,KeyError,TypeError) as exc:
            if not allow_invalid: raise
            structured=None; repaired=False; error=f"{type(exc).__name__}: {exc}"
    metadata=dict(response.metadata or {})
    if model_name: metadata["model_name"]=model_name
    if model_revision: metadata["model_revision"]=model_revision
    return archive_payload(request,response.text,structured,response.latency_seconds,response.input_tokens,
                           response.output_tokens,metadata,first_pass_valid,repaired,
                           final_executable=structured is not None,error=error)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--request",type=Path,required=True)
    p.add_argument("--backend",choices=("command","local"),required=True)
    p.add_argument("--command"); p.add_argument("--model-path")
    p.add_argument("--model-name",choices=SUPPORTED_MODELS,default=DEFAULT_MODEL)
    p.add_argument("--model-revision"); p.add_argument("--timeout",type=float,default=300)
    p.add_argument("--max-new-tokens",type=int,default=2048); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    if args.backend=="command" and not args.command: p.error("--command required for command")
    if args.backend=="local" and not args.model_path: p.error("--model-path required for local")
    archive=execute(json.loads(args.request.read_text()),make_backend(args),args.model_name,args.model_revision)
    write_json(args.output,archive)
if __name__=="__main__": main()
