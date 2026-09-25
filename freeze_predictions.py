#!/usr/bin/env python3
"""Refit on training data, execute the frozen paper program, and freeze predictions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from common import quantile_decode,sha256,write_json
from core_model import fit_model,predict
from data_io import load_evaluation_metadata,load_split,load_tokens,load_training_labels
from evidence import HeadConfig,fit_predict
from program_execution import apply_correction,compose_from_ranks,correction_features,population_ranks

def execute_program(base_rank,source_rank,program):
    composed,evidence=compose_from_ranks(base_rank,source_rank,program)
    features=correction_features(composed,base_rank,source_rank,program)
    corrected,delta=apply_correction(composed,features,program,program.get("correction_model"))
    return corrected,composed,evidence,delta

def main():
    p=argparse.ArgumentParser(); p.add_argument("--split",type=Path,required=True)
    p.add_argument("--train-labels",type=Path,required=True); p.add_argument("--eval-metadata",type=Path,required=True)
    p.add_argument("--i3d",type=Path,required=True); p.add_argument("--swin",type=Path,required=True); p.add_argument("--pose",type=Path,required=True)
    p.add_argument("--family",type=Path,required=True); p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--device",default="cuda:0"); args=p.parse_args()
    manifest,train_ids,test_ids=load_split(args.split); all_ids=train_ids+test_ids
    target,train_difficulty=load_training_labels(args.train_labels,train_ids)
    test_difficulty=load_evaluation_metadata(args.eval_metadata,test_ids)
    tokens=load_tokens(args.i3d,args.swin,args.pose,all_ids); n_train=len(train_ids)
    train_index=np.arange(n_train); test_index=np.arange(n_train,len(all_ids))
    selection=json.loads(args.selection.read_text()); selected=selection["selected_program"]
    selected_nonlanguage=selection["selected_nonlanguage_program"]
    if selection["integrity"]["family_sha256"]!=sha256(args.family): raise ValueError("family hash mismatch")
    difficulty=np.r_[train_difficulty,test_difficulty]; component=np.r_[target/train_difficulty,np.zeros(len(test_ids))]
    base_features=tokens["i3d"].mean(1).astype("f4"); device=torch.device(args.device); base_members=[]
    for seed in (20260911,20260929):
        model,bounds,groups=fit_model(base_features,component,difficulty,train_index,seed,device)
        base_members.append(predict(model,bounds,groups,base_features,component,difficulty,test_index,seed,device))
    base_raw=np.mean(base_members,axis=0); source_raw={}
    for source,source_tokens in tokens.items():
        row=selection["final_head_configs"][source]; config=HeadConfig(row["view"],float(row["alpha"]))
        source_raw[source]=fit_predict(source_tokens,target,train_index,test_index,config)
    base_rank,source_rank=population_ranks(base_raw,source_raw)
    selected_rank,selected_uncorrected,selected_evidence,selected_delta=execute_program(base_rank,source_rank,selected)
    nonlanguage_rank,_,_,_=execute_program(base_rank,source_rank,selected_nonlanguage)
    candidate_rows=selection["candidates"]
    worst=min(candidate_rows,key=lambda row:(row["M_p"],row["G_p"],-row["fixed_index"]))
    worst_rank,_,_,_=execute_program(base_rank,source_rank,worst)
    generator=np.random.default_rng(20260905)
    random_rows=[candidate_rows[int(index)] for index in generator.choice(len(candidate_rows),size=min(64,len(candidate_rows)),replace=False)]
    random_ids=[row["candidate_id"] for row in random_rows]
    random_rank=np.stack([execute_program(base_rank,source_rank,row)[0] for row in random_rows])
    ranks={"CirScore":selected_rank,"w_o_LLM":nonlanguage_rank,"w_o_NS":base_rank,
           "w_o_Correction":selected_uncorrected,"worst_legal":worst_rank}
    scores={name:(base_raw if name=="w_o_NS" else quantile_decode(value,target)) for name,value in ranks.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,sample_id=np.asarray(test_ids),base_member_raw=np.stack(base_members),
                        trace_base_rank=base_rank,trace_i3d_rank=source_rank["i3d"],trace_swin_rank=source_rank["swin"],
                        trace_pose_rank=source_rank["pose"],trace_selected_evidence=selected_evidence,
                        trace_composed_rank=selected_uncorrected,trace_correction_delta=selected_delta,
                        trace_corrected_rank=selected_rank,random_candidate_id=np.asarray(random_ids),random_rank=random_rank,
                        random_score=np.stack([quantile_decode(row,target) for row in random_rank]),
                        **{f"rank_{name}":value for name,value in ranks.items()},
                        **{f"score_{name}":value for name,value in scores.items()})
    write_json(args.audit,{"status":"FROZEN_WITHOUT_TEST_LABEL_FILE","test_label_file_opened":False,
                           "selected_program":selected,"selected_nonlanguage_program":selected_nonlanguage,
                           "control_applicability":{"w_o_Correction":bool(selected["correction_enabled"])},
                           "worst_training_program":worst,
                           "execution_trace_fields":["base_rank","i3d_rank","swin_rank","pose_rank","selected_evidence",
                                                     "composed_rank","correction_delta","corrected_rank","score_CirScore"],
                           "n_train":len(train_ids),"n_test":len(test_ids),"split_sha256":sha256(args.split),
                           "train_labels_sha256":sha256(args.train_labels),"eval_metadata_sha256":sha256(args.eval_metadata),
                           "selection_sha256":sha256(args.selection),"prediction_sha256":sha256(args.output)})

if __name__=="__main__": main()
