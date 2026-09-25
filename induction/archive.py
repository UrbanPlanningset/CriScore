#!/usr/bin/env python3
"""Create a sanitized, hash-linked induction archive."""
from __future__ import annotations
import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import canonical_hash,write_json

def text_hash(text: str):
    return hashlib.sha256(text.encode()).hexdigest()

def archive_payload(request,response_text,structured,latency,input_tokens,output_tokens,metadata,first_pass_valid,repair_applied,final_executable=True,error=None):
    return {"schema":"cirscore-induction-archive-v1","request_sha256":canonical_hash(request),
            "response_sha256":text_hash(response_text),
            "structured_output_sha256":None if structured is None else canonical_hash(structured),
            "usage":{"input_tokens":input_tokens,"output_tokens":output_tokens,
                     "total_tokens":None if input_tokens is None or output_tokens is None else input_tokens+output_tokens},
            "latency_seconds":latency,"first_pass_valid":first_pass_valid,"repair_applied":repair_applied,
            "final_executable":final_executable,"error":error,
            "backend_metadata":metadata or {},"structured_output":structured}
