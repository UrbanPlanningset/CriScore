#!/usr/bin/env python3
"""Open the physically separate test-label file only after prediction freezing."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from common import sha256,write_json
from data_io import indexed_npz,load_split,ordered

def metrics(target,prediction):
    error=target-prediction; denominator=np.sum((target-target.mean())**2)
    if denominator<=0: raise ValueError("constant test target")
    srcc=float(spearmanr(target,prediction).statistic)
    if not np.isfinite(srcc) or not np.isfinite(error).all(): raise ValueError("non-finite or undefined test metric")
    return {"srcc_percent":100*srcc,
            "nmse":float(np.sum(error**2)/denominator),"mae":float(np.mean(np.abs(error))),
            "rmse":float(np.sqrt(np.mean(error**2)))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--split",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True); p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--test-labels",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    audit=json.loads(args.audit.read_text())
    if audit["status"]!="FROZEN_WITHOUT_TEST_LABEL_FILE" or audit["test_label_file_opened"] is not False:
        raise ValueError("invalid pre-label freeze audit")
    if audit["prediction_sha256"]!=sha256(args.predictions) or audit["split_sha256"]!=sha256(args.split):
        raise ValueError("frozen artifact hash mismatch")
    _,_,test_ids=load_split(args.split); frozen=np.load(args.predictions,allow_pickle=False)
    if list(map(str,frozen["sample_id"]))!=test_ids: raise ValueError("test order mismatch")
    labels,lookup=indexed_npz(args.test_labels,("sample_id","target"))
    target=ordered(labels["target"],lookup,test_ids).astype(float)
    names=["CirScore","w_o_LLM","w_o_NS","worst_legal"]
    not_applicable=[]
    if audit.get("control_applicability",{}).get("w_o_Correction",False): names.append("w_o_Correction")
    else: not_applicable.append("w_o_Correction")
    result={name:metrics(target,frozen[f"score_{name}"].astype(float)) for name in names}
    random=[metrics(target,row.astype(float)) for row in frozen["random_score"]]
    q95=float(np.quantile([row["srcc_percent"] for row in random],.95))
    write_json(args.output,{"metrics":result,"not_applicable_controls":not_applicable,
                            "random_program_q95_srcc_percent":q95,
                            "CirScore_minus_random_q95_pp":result["CirScore"]["srcc_percent"]-q95,
                            "test_labels_sha256":sha256(args.test_labels),"predictions_sha256":sha256(args.predictions)})

if __name__=="__main__": main()
