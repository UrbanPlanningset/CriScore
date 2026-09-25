#!/usr/bin/env python3
"""Schemas and validators for paper-defined induction inputs and outputs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contract import CONCEPTS,CONSTRAINTS,SOURCES

@dataclass(frozen=True)
class InductionRequest:
    task_description: str
    evidence_specification: dict[str,Any]
    training_diagnostics: dict[str,Any]
    output_schema: dict[str,Any]

REQUIRED_OUTPUT_FIELDS=("concepts","constraints","registered_sources","relation_operator","fallback","open_fields","bindings","relations")
COMPOSITION_RELATIONS=("EQUAL_PRIORITY_SOURCE_COMPOSE","PRIORITY_WEIGHTED_SOURCE_COMPOSE")
REQUIRED_RELATIONS=("AVERAGE_TIES_RANK_NORMALIZE","BLEND_WITH_BASE","TRAIN_LABEL_QUANTILE_DECODE")
ALLOWED_RELATIONS=REQUIRED_RELATIONS+COMPOSITION_RELATIONS

def output_schema():
    return {
      "type":"object","required":list(REQUIRED_OUTPUT_FIELDS),
      "additionalProperties":False,
      "properties":{
        "concepts":{"type":"array","minItems":len(CONCEPTS),"maxItems":len(CONCEPTS),"uniqueItems":True,
                    "items":{"type":"string","enum":list(CONCEPTS)}},
        "constraints":{"type":"array","minItems":len(CONSTRAINTS),"maxItems":len(CONSTRAINTS),"uniqueItems":True,
                       "items":{"type":"string","enum":list(CONSTRAINTS)}},
        "registered_sources":{"type":"array","minItems":len(SOURCES),"maxItems":len(SOURCES),"uniqueItems":True,
                              "items":{"type":"string","enum":list(SOURCES)}},
        "relation_operator":{"const":"rank_blend"},"fallback":{"const":"base"},
        "open_fields":{"type":"array","minItems":2,"maxItems":2,"uniqueItems":True,
                       "items":{"type":"string","enum":["active_source_subset","blend_strength"]}},
        "bindings":{"type":"array","minItems":len(CONCEPTS),"maxItems":len(CONCEPTS),
                    "items":{"type":"object","required":["concept","compatible_sources"],"additionalProperties":False,
                             "properties":{"concept":{"type":"string","enum":list(CONCEPTS)},
                                           "compatible_sources":{"type":"array","minItems":1,"uniqueItems":True,
                                                                 "items":{"type":"string","enum":list(SOURCES)}}}}},
        "relations":{"type":"array","minItems":4,"uniqueItems":True,
                     "items":{"type":"string","enum":list(ALLOWED_RELATIONS)}},
      },
    }

def validate_request(payload):
    for key in ("task_description","evidence_specification","training_diagnostics","output_schema"):
        if key not in payload: raise ValueError(f"request missing {key}")
    if not str(payload["task_description"]).strip(): raise ValueError("empty task description")
    sources=tuple(payload["evidence_specification"].get("sources",()))
    if sources!=SOURCES: raise ValueError("request evidence vocabulary mismatch")
    return payload

def validate_output(payload):
    missing=set(REQUIRED_OUTPUT_FIELDS)-set(payload)
    if missing: raise ValueError(f"induction output missing {sorted(missing)}")
    if set(payload["concepts"])!=set(CONCEPTS): raise ValueError("concept set mismatch")
    if set(payload["constraints"])!=set(CONSTRAINTS): raise ValueError("constraint set mismatch")
    if tuple(payload["registered_sources"])!=SOURCES: raise ValueError("source set mismatch")
    if payload["relation_operator"]!="rank_blend" or payload["fallback"]!="base": raise ValueError("relation mismatch")
    if set(payload["open_fields"])!={"active_source_subset","blend_strength"}: raise ValueError("open fields mismatch")
    binding_concepts=[row["concept"] for row in payload["bindings"]]
    if len(binding_concepts)!=len(CONCEPTS) or set(binding_concepts)!=set(CONCEPTS):
        raise ValueError("every concept requires exactly one binding declaration")
    for row in payload["bindings"]:
        if not set(row["compatible_sources"]).issubset(SOURCES) or not row["compatible_sources"]:
            raise ValueError("invalid compatible source binding")
    if not set(REQUIRED_RELATIONS).issubset(set(payload["relations"])): raise ValueError("executable relations incomplete")
    if len(set(COMPOSITION_RELATIONS).intersection(payload["relations"]))!=1:
        raise ValueError("source-composition relation missing")
    return payload

def canonical_request_text(payload):
    validate_request(payload)
    instruction=("Return exactly one JSON object matching output_schema. Copy every fixed concept, constraint, "
                 "registered source, open field, and required relation exactly as enumerated. Choose exactly one "
                 "source-composition relation. Declare exactly one nonempty compatible_sources binding for every "
                 "concept. Do not add prose or fields outside the schema.")
    return instruction+"\n"+json.dumps(payload,sort_keys=True,separators=(",",":"))
