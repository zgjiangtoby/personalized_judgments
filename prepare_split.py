"""Construct crossed person/text splits and select two anchor responses per rater."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--rater-partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    frame = pd.read_parquet(args.ratings,
                            columns=["comment_id","annotator_id","text","hatespeech"])
    people = pd.read_csv(args.rater_partition)
    frame = frame.rename(columns={"comment_id":"item_id","annotator_id":"rater_id"})
    assert len(frame) == 135556 and not frame.duplicated(["item_id","rater_id"]).any()
    assert frame.hatespeech.isin([0,1,2]).all() and frame.text.notna().all()
    assert people.rater_id.is_unique and set(frame.rater_id) == set(people.rater_id)
    items = frame[["item_id","text"]].drop_duplicates().sort_values("item_id").reset_index(drop=True)
    assert items.item_id.is_unique and items.text.is_unique and len(items) == 39565
    rng = np.random.default_rng(20261009)
    people["rater_split"] = "test"
    discovery = people.index[people.phase.eq("discovery")].to_numpy()
    order = rng.permutation(discovery)
    stop = int(np.floor(0.8*len(order)))
    people.loc[order[:stop],"rater_split"] = "train"
    people.loc[order[stop:],"rater_split"] = "validation"
    sizes = frame.groupby("item_id").size()
    anchors = set(sizes.index[sizes.gt(50)])
    assert len(anchors) == 70
    candidates = rng.permutation(items.loc[~items.item_id.isin(anchors),"item_id"].to_numpy())
    first, second = int(0.6*len(candidates)), int(0.8*len(candidates))
    text_split = {int(i): "anchor" for i in anchors}
    for split, ids in [("train",candidates[:first]),("validation",candidates[first:second]),("test",candidates[second:])]:
        text_split.update({int(i):split for i in ids})
    items["text_split"] = items.item_id.map(text_split)
    frame = frame.merge(people[["rater_id","reconstructed_gender","rater_split"]],
                        on="rater_id",validate="many_to_one")
    frame["text_split"] = frame.item_id.map(text_split)
    frame["calibration"] = False
    for rater, group in frame[frame.text_split.eq("anchor")].sort_values(["rater_id","item_id"]).groupby("rater_id",sort=True):
        chosen = rng.choice(group.index.to_numpy(),size=min(2,len(group)),replace=False)
        frame.loc[chosen,"calibration"] = True
    assert not frame.loc[frame.text_split.eq("test"),"calibration"].any()
    assert not frame.loc[frame.rater_split.eq("test") & frame.text_split.eq("test"),"item_id"].isin(anchors).any()
    counts = frame.groupby(["rater_split","text_split"],sort=True).agg(
        ratings=("rater_id","size"), raters=("rater_id","nunique"), texts=("item_id","nunique"))
    calibration = frame[frame.calibration].groupby("rater_id").size()
    test_raters = frame.loc[frame.rater_split.eq("test") & frame.text_split.eq("test"),"rater_id"].unique()
    cal_support = pd.Series(test_raters).map(calibration).fillna(0).astype(int).value_counts().sort_index().to_dict()
    args.output.mkdir(parents=True)
    people.to_csv(args.output/"raters.csv",index=False)
    items[["item_id","text_split"]].to_csv(args.output/"items.csv",index=False)
    frame.drop(columns="text").to_parquet(args.output/"responses.parquet",index=False)
    train_ids = items.loc[items.text_split.eq("train"),"item_id"].to_numpy()
    pilot_ids = set(rng.choice(train_ids,size=32,replace=False).tolist())
    with (args.output/"texts.jsonl").open("x") as full, (args.output/"pilot.jsonl").open("x") as pilot:
        for row in items.itertuples(index=False):
            line = json.dumps({"id":str(row.item_id),"text":row.text},ensure_ascii=False)+"\n"
            full.write(line)
            if row.item_id in pilot_ids:
                pilot.write(line)
    summary = {"cells":counts.reset_index().to_dict(orient="records"),
               "new_person_new_text_raters_by_anchor_calibration_count":{str(k):int(v) for k,v in cal_support.items()},
               "anchors":len(anchors),"unique_texts":len(items),"pilot_training_texts":32,
               "partition_uses_outcome_values":False,"seed":20261009}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
