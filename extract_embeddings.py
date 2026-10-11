"""Extract attention-masked means of final-layer text representations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from model_runtime import load_local, metadata

MODULE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def embed_batch(model,tokenizer,sequences):
    batch = tokenizer.pad({"input_ids":sequences},padding=True,return_tensors="pt").to("cuda")
    positions = (batch.attention_mask.cumsum(-1)-1).clamp_min(0)
    hidden = model.model(**batch,position_ids=positions,use_cache=False).last_hidden_state.float()
    weights = batch.attention_mask[...,None]
    return (hidden*weights).sum(1)/weights.sum(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16","float32"), default="bfloat16")
    parser.add_argument("--check-batch-equivalence",action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available() and not args.output.exists() and args.batch_size > 0 and args.max_tokens > 0
    input_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines() if line.strip()]
    assert rows and len({row["id"] for row in rows}) == len(rows)
    tokenizer, model = load_local(args.model,args.dtype)
    sequences = [tokenizer.encode(row["text"],add_special_tokens=False) for row in rows]
    assert all(sequences)
    original_lengths = list(map(len,sequences))
    sequences = [sequence[:args.max_tokens] for sequence in sequences]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32,
                                      shape=(len(rows),model.config.hidden_size))
    meta = metadata(args,tokenizer)
    meta["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    meta["runtime_source_sha256"]["extract_embeddings.py"] = MODULE_SOURCE_SHA256
    meta.update(pooling="final-layer arithmetic mean over retained content tokens", template="raw text; no chat or special tokens",
                truncation=f"retain first {args.max_tokens} tokens", rows_truncated=sum(n>args.max_tokens for n in original_lengths))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if args.check_batch_equivalence:
            probes = sequences[:min(args.batch_size,len(sequences))]
            batched = embed_batch(model,tokenizer,probes)
            single = torch.cat([embed_batch(model,tokenizer,[sequence]) for sequence in probes])
            difference = (batched-single).norm(dim=1)/single.norm(dim=1).clamp_min(1e-12)
            meta["batch_vector_max_relative_l2_difference"] = float(difference.max())
            meta["batch_equivalence_probe_size"] = len(probes)
            assert float(difference.max()) < (1e-4 if args.dtype=="float32" else 0.02)
        for offset in range(0,len(rows),args.batch_size):
            pooled = embed_batch(model,tokenizer,sequences[offset:offset+args.batch_size])
            assert torch.isfinite(pooled).all()
            vectors[offset:offset+len(pooled)] = pooled.cpu().numpy()
            if offset == 0 or (offset//args.batch_size+1)%50 == 0:
                print(json.dumps({"completed":min(offset+args.batch_size,len(rows)),"total":len(rows),
                                  "elapsed_seconds":time.monotonic()-start}),flush=True)
    vectors.flush()
    with args.output.with_suffix(".items.jsonl").open("x") as stream:
        for row, original, kept in zip(rows,original_lengths,map(len,sequences)):
            stream.write(json.dumps({"id":row["id"],"original_tokens":original,"retained_tokens":kept})+"\n")
    torch.cuda.synchronize()
    meta.update(rows=len(rows), dimensions=model.config.hidden_size, elapsed_seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), status="complete")
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)


if __name__ == "__main__":
    main()
