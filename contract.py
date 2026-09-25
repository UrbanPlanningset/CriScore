#!/usr/bin/env python3
"""Typed paper contract for the CirScore induction and execution interface."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Tuple

CONCEPTS: Tuple[str,...]=(
    "EARLY_STATE","MIDDLE_TRANSITION","LATE_EXECUTION",
    "TEMPORAL_COHERENCE","ACTION_COMPARABILITY","DIFFICULTY_MATCHED_PAIR",
)
CONSTRAINTS: Tuple[str,...]=(
    "ORDERED_PROGRESS","COHERENCE_SUPPORTS_QUALITY","COMPARE_WITHIN_ACTION",
    "COMPARE_NEAR_DIFFICULTY","ABSTAIN_TO_CORE",
)
SOURCES: Tuple[str,...]=("i3d","swin","pose")
BETAS: Tuple[float,...]=(0.25,0.5,0.75,1.0)
TEMPERATURES: Tuple[float,...]=(0.5,1.0,2.0)
CORRECTION_BOUND=0.10
CORRECTION_STRENGTH=1.0
MAX_CANDIDATES=64

@dataclass(frozen=True)
class InductionRecord:
    concepts: Tuple[str,...]
    constraints: Tuple[str,...]
    registered_sources: Tuple[str,...]
    relation_operator: str
    fallback: str
    open_fields: Tuple[str,...]
    bindings: Tuple[Tuple[str,Tuple[str,...]],...]
    relations: Tuple[str,...]

@dataclass(frozen=True)
class Program:
    fixed_index: int
    candidate_id: str
    concepts: Tuple[str,...]
    bindings: Tuple[Tuple[str,str],...]
    relations: Tuple[str,...]
    sources: Tuple[str,...]
    source_weights: Tuple[float,...]
    temperature: float
    beta: float
    correction_enabled: bool
    correction_coordinate: str
    correction_bound: float
    correction_strength: float
    fallback: str

def induction_from_dict(payload: dict) -> InductionRecord:
    return InductionRecord(
        concepts=tuple(payload["concepts"]),
        constraints=tuple(payload["constraints"]),
        registered_sources=tuple(payload["registered_sources"]),
        relation_operator=str(payload["relation_operator"]),
        fallback=str(payload["fallback"]),
        open_fields=tuple(payload["open_fields"]),
        bindings=tuple((str(row["concept"]),tuple(map(str,row["compatible_sources"]))) for row in payload["bindings"]),
        relations=tuple(map(str,payload["relations"])),
    )

def validate_induction(record: InductionRecord) -> None:
    if set(record.concepts)!=set(CONCEPTS): raise ValueError("six-concept induction mismatch")
    if set(record.constraints)!=set(CONSTRAINTS): raise ValueError("five-constraint induction mismatch")
    if tuple(record.registered_sources)!=SOURCES: raise ValueError("evidence vocabulary mismatch")
    if record.relation_operator!="rank_blend" or record.fallback!="base":
        raise ValueError("unsupported induced relation")
    if set(record.open_fields)!={"active_source_subset","blend_strength"}:
        raise ValueError("paper-open calibration fields mismatch")
    binding_concepts=[concept for concept,_ in record.bindings]
    if len(binding_concepts)!=len(CONCEPTS) or set(binding_concepts)!=set(CONCEPTS):
        raise ValueError("every concept requires exactly one induced binding")
    for _,sources in record.bindings:
        if not sources or not set(sources).issubset(SOURCES): raise ValueError("invalid induced binding")
    required={"AVERAGE_TIES_RANK_NORMALIZE","BLEND_WITH_BASE","TRAIN_LABEL_QUANTILE_DECODE"}
    if not required.issubset(record.relations): raise ValueError("induced relations are incomplete")
    composition={"EQUAL_PRIORITY_SOURCE_COMPOSE","PRIORITY_WEIGHTED_SOURCE_COMPOSE"}.intersection(record.relations)
    if len(composition)!=1: raise ValueError("exactly one induced composition relation is required")

def to_dict(value):
    return asdict(value)
