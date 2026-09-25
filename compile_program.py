#!/usr/bin/env python3
"""Compile language-induced and matched non-language program families."""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
from common import canonical_hash,write_json
from contract import (BETAS,CONCEPTS,CORRECTION_BOUND,CORRECTION_STRENGTH,MAX_CANDIDATES,
                      Program,SOURCES,TEMPERATURES,induction_from_dict,to_dict,validate_induction)

RELATIONS=("AVERAGE_TIES_RANK_NORMALIZE","PRIORITY_WEIGHTED_SOURCE_COMPOSE","BLEND_WITH_BASE",
           "OPTIONAL_BOUNDED_RANK_CORRECTION","TRAIN_LABEL_QUANTILE_DECODE")

def _make_program(index,sources,weights,temperature,beta,correction,bindings,relations,fallback):
    identity={"sources":sources,"weights":weights,"temperature":temperature,"beta":beta,
              "correction":correction,"bindings":bindings,"relations":relations,"fallback":fallback}
    return Program(index,canonical_hash(identity),CONCEPTS,tuple(bindings),tuple(relations),tuple(sources),
                   tuple(map(float,weights)),float(temperature),float(beta),bool(correction),"rank",
                   CORRECTION_BOUND,CORRECTION_STRENGTH,fallback)

def _deduplicate(rows):
    seen=set(); output=[]
    for row in rows:
        if row.candidate_id in seen: continue
        seen.add(row.candidate_id); output.append(row)
    return output

def compile_language_family(induction):
    compatibility={concept:set(sources) for concept,sources in induction.bindings}; proposed=[]
    for size in range(1,len(induction.registered_sources)+1):
        for sources in itertools.combinations(induction.registered_sources,size):
            selected=set(sources)
            if any(not (compatibility[concept]&selected) for concept in CONCEPTS): continue
            bindings=tuple((concept,source) for concept in CONCEPTS for source in sources if source in compatibility[concept])
            counts=np.asarray([sum(source in compatibility[concept] for concept in CONCEPTS) for source in sources],float)
            if np.any(counts==0): continue
            weights=tuple((counts/counts.sum()).tolist())
            relation_rows=list(induction.relations)
            if not np.allclose(weights,np.full(len(weights),1.0/len(weights))):
                relation_rows=["PRIORITY_WEIGHTED_SOURCE_COMPOSE" if row=="EQUAL_PRIORITY_SOURCE_COMPOSE" else row for row in relation_rows]
            relations=tuple(dict.fromkeys(tuple(relation_rows)+("OPTIONAL_BOUNDED_RANK_CORRECTION",)))
            for temperature,beta,correction in itertools.product(TEMPERATURES,BETAS,(False,True)):
                proposed.append(_make_program(0,sources,weights,temperature,beta,correction,bindings,
                                              relations,induction.fallback))
    proposed=_deduplicate(proposed)
    proposed.sort(key=lambda row:row.candidate_id)
    return [Program(index,row.candidate_id,row.concepts,row.bindings,row.relations,row.sources,row.source_weights,
                    row.temperature,row.beta,row.correction_enabled,row.correction_coordinate,row.correction_bound,
                    row.correction_strength,row.fallback)
            for index,row in enumerate(proposed[:MAX_CANDIDATES])]

def _simplex_weights(size,resolution=10):
    if size==1:
        yield (1.0,); return
    for units in itertools.product(range(resolution+1),repeat=size):
        if sum(units)==resolution: yield tuple(value/resolution for value in units)

def compile_nonlanguage_family(seed=20260905):
    subsets=[sources for size in range(1,len(SOURCES)+1) for sources in itertools.combinations(SOURCES,size)]
    weight_options={size:list(_simplex_weights(size)) for size in range(1,len(SOURCES)+1)}
    generator=np.random.default_rng(seed); chosen=[]; seen=set(); attempts=0
    while len(chosen)<MAX_CANDIDATES and attempts<10000:
        attempts+=1; sources=subsets[int(generator.integers(len(subsets)))]
        options=weight_options[len(sources)]; weights=options[int(generator.integers(len(options)))]
        temperature=TEMPERATURES[int(generator.integers(len(TEMPERATURES)))]
        beta=BETAS[int(generator.integers(len(BETAS)))]; correction=bool(generator.integers(2))
        bindings=tuple((concept,source) for concept in CONCEPTS for source in sources)
        row=_make_program(0,sources,weights,temperature,beta,correction,bindings,RELATIONS,"base")
        if row.candidate_id in seen: continue
        seen.add(row.candidate_id); chosen.append(row)
    if len(chosen)<MAX_CANDIDATES: raise RuntimeError("could not sample 64 distinct non-language programs")
    return [Program(index,row.candidate_id,row.concepts,row.bindings,row.relations,row.sources,row.source_weights,
                    row.temperature,row.beta,row.correction_enabled,row.correction_coordinate,row.correction_bound,
                    row.correction_strength,row.fallback)
            for index,row in enumerate(chosen)]

def compile_family(induction_payload):
    induction=induction_from_dict(induction_payload); validate_induction(induction)
    programs=compile_language_family(induction); nonlanguage=compile_nonlanguage_family()
    if not programs: raise ValueError("language induction produced no legal executable program")
    return induction,programs,nonlanguage

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--induction",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    payload=json.loads(args.induction.read_text())
    if "structured_output" in payload: payload=payload["structured_output"]
    induction,programs,nonlanguage=compile_family(payload)
    write_json(args.output,{"schema":"cirscore-family-v3","induction":to_dict(induction),"candidate_count":len(programs),
                            "nonlanguage_candidate_count":len(nonlanguage),
                            "selection_rule":"argmax_lex(M_p,G_p,-fixed_index)","positive_gain_gate":False,
                            "programs":[to_dict(p) for p in programs],
                            "nonlanguage_programs":[to_dict(p) for p in nonlanguage]})

if __name__=="__main__": main()
