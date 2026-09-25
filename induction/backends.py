#!/usr/bin/env python3
"""External-command and optional local-model induction backends."""
from __future__ import annotations
from dataclasses import dataclass
import json,subprocess,time

@dataclass(frozen=True)
class BackendResponse:
    text: str
    latency_seconds: float
    input_tokens: int|None=None
    output_tokens: int|None=None
    metadata: dict|None=None

class CommandBackend:
    def __init__(self,command: list[str],timeout: float): self.command=command; self.timeout=timeout
    def generate(self,prompt: str) -> BackendResponse:
        started=time.perf_counter()
        completed=subprocess.run(self.command,input=prompt,text=True,capture_output=True,timeout=self.timeout,check=True,shell=False)
        text=completed.stdout; input_tokens=output_tokens=None; metadata={"backend":"command","returncode":completed.returncode}
        try:
            envelope=json.loads(text)
            if isinstance(envelope,dict) and "response_text" in envelope:
                text=str(envelope["response_text"]); usage=envelope.get("usage",{})
                input_tokens=usage.get("input_tokens"); output_tokens=usage.get("output_tokens")
                metadata.update(envelope.get("metadata",{}))
        except json.JSONDecodeError:
            pass
        return BackendResponse(text,time.perf_counter()-started,input_tokens,output_tokens,metadata)

class LocalBackend:
    def __init__(self,model_path: str,max_new_tokens: int=2048):
        try:
            from transformers import AutoModelForCausalLM,AutoTokenizer
        except ImportError as exc: raise RuntimeError("local backend requires transformers") from exc
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,device_map="auto")
        self.max_new_tokens=max_new_tokens
    def generate(self,prompt: str) -> BackendResponse:
        import torch
        started=time.perf_counter(); inputs=self.tokenizer(prompt,return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        generated=output[0,inputs["input_ids"].shape[1]:]
        return BackendResponse(self.tokenizer.decode(generated,skip_special_tokens=True),time.perf_counter()-started,
                               int(inputs["input_ids"].numel()),int(generated.numel()),{"backend":"local"})
