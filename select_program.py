#!/usr/bin/env python3
"""Apply cross-fitted calibration and the exact paper selection rule."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from common import rank01,rho,sha256,write_json
from evidence import fit_predict,select_head_config
from program_execution import apply_correction,compose_from_ranks,correction_features,fit_correction

def foldwise_rank(values,fold_id,folds):
    output=np.full(len(values),np.nan)
    for fold in folds:
        mask=fold_id==fold; output[mask]=rank01(values[mask])
    return output

def evaluate_family(programs,target,fold_id,folds,base_rank,source_rank):
    target_rank=foldwise_rank(target,fold_id,folds); rows=[]
    for program in programs:
        composed=np.full(len(target),np.nan); evidence=np.full(len(target),np.nan)
        for fold in folds:
            mask=fold_id==fold
            local_sources={source:values[mask] for source,values in source_rank.items()}
            composed[mask],evidence[mask]=compose_from_ranks(base_rank[mask],local_sources,program)
        features=correction_features(composed,base_rank,source_rank,program)
        corrected=np.array(composed,copy=True)
        if program.get("correction_enabled",False):
            for held_fold in folds:
                fit=fold_id!=held_fold; held=fold_id==held_fold
                model=fit_correction(features[fit],target_rank[fit]-composed[fit])
                corrected[held],_=apply_correction(composed[held],features[held],program,model)
            final_correction=fit_correction(features,target_rank-composed)
        else:
            final_correction=None
        fold_srcc=[rho(target[fold_id==fold],corrected[fold_id==fold]) for fold in folds]
        fold_gain=[score-rho(target[fold_id==fold],base_rank[fold_id==fold]) for fold,score in zip(folds,fold_srcc)]
        rows.append({**program,"fold_srcc":fold_srcc,"fold_gain":fold_gain,
                     "M_p":float(np.mean(fold_srcc)),"G_p":float(np.min(fold_gain)),
                     "correction_model":final_correction})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--family",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    z=np.load(args.bundle,allow_pickle=False); target=z["target"].astype(float); fold_id=z["fold_id"].astype(int)
    base=z["base_oof"].astype(float); tokens={name:z[f"{name}_tokens"].astype("f4") for name in ("i3d","swin","pose")}
    family=json.loads(args.family.read_text())
    if family["positive_gain_gate"] is not False: raise ValueError("compiled family mismatch")
    programs=family["programs"]; nonlanguage=family["nonlanguage_programs"]
    if len(programs)!=family["candidate_count"] or len(nonlanguage)!=family["nonlanguage_candidate_count"]:
        raise ValueError("candidate count mismatch")
    folds=sorted(map(int,np.unique(fold_id))); fold_head_configs={}
    source_oof={source:np.full(len(target),np.nan) for source in tokens}
    for held_fold in folds:
        fit_index=np.flatnonzero(fold_id!=held_fold); held_index=np.flatnonzero(fold_id==held_fold); source_prediction={}
        fold_head_configs[str(held_fold)]={}
        for source in tokens:
            config=select_head_config(tokens[source],target,fold_id,excluded_fold=held_fold)
            fold_head_configs[str(held_fold)][source]={"view":config.view,"alpha":config.alpha}
            source_prediction[source]=fit_predict(tokens[source],target,fit_index,held_index,config)
            source_oof[source][held_index]=source_prediction[source]
    if any(not np.isfinite(values).all() for values in source_oof.values()): raise RuntimeError("incomplete source OOF predictions")
    base_rank=foldwise_rank(base,fold_id,folds)
    source_rank={source:foldwise_rank(values,fold_id,folds) for source,values in source_oof.items()}
    rows=evaluate_family(programs,target,fold_id,folds,base_rank,source_rank)
    nonlanguage_rows=evaluate_family(nonlanguage,target,fold_id,folds,base_rank,source_rank)
    selected=max(rows,key=lambda row:(row["M_p"],row["G_p"],-row["fixed_index"]))
    selected_nonlanguage=max(nonlanguage_rows,key=lambda row:(row["M_p"],row["G_p"],-row["fixed_index"]))
    final_head_configs={}
    for source in tokens:
        config=select_head_config(tokens[source],target,fold_id)
        final_head_configs[source]={"view":config.view,"alpha":config.alpha}
    write_json(args.output,{"schema":"cirscore-selection-v3","selection_rule":"argmax_lex(M_p,G_p,-fixed_index)",
                            "positive_gain_gate":False,"selected_program":selected,
                            "selected_nonlanguage_program":selected_nonlanguage,
                            "final_head_configs":final_head_configs,"fold_head_configs":fold_head_configs,
                            "candidates":rows,"nonlanguage_candidates":nonlanguage_rows,
                            "integrity":{"bundle_sha256":sha256(args.bundle),"family_sha256":sha256(args.family)}})

if __name__=="__main__": main()
