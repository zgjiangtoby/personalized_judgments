# Personalized judgments

Companion code for *Attributing Personalization Gains with Frozen LLM Representations*, manuscript version 11 October 2026 (eight-page main text). Run the commands from this directory.

Ordinal hate-speech judgments with crossed person/text holdouts, anchor calibration and matched personal text directions. The lexical control uses the same calibration budgets.

## Environment

Run the commands below in Bash. Use Python 3.11 and a CUDA GPU with enough memory for FP32 inference; the 8B runs used GPUs reporting 49,140 MiB. Keep the batch settings below when comparing the reported results.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 pandas==2.3.3 scipy==1.16.3 \
  transformers==4.57.2 tokenizers==0.22.1 accelerate==1.14.0 \
  safetensors==0.7.0 sentencepiece==0.2.1 huggingface-hub==0.36.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data models work
```

Use new output directories for a run. Inference writes score metadata beside the score file; analyses use those files and intermediate outputs from the earlier commands. Raw texts, labels, model weights and generated results are not included. Obtain the sources below under their own terms and retain their notices.

```bash
python -m pip install scikit-learn==1.7.2 pyarrow==24.0.0
```

## Data and splits

Four released sources are required. The official dataset provides the text and 0/1/2 outcome for the LLM analysis. Three research copies reconstruct the available women/men grouping: original ratings, a social-group carrier, and a smaller copy with self-reported fields. The join checks the released keys and outcome/text agreement; missing demographic codes remain missing. This grouping is not a complete native demographic release.

```bash
hf download ucberkeley-dlab/measuring-hate-speech --repo-type dataset \
  --revision 5468f6e118396646b02a2f691e771f6b6d9502ea --local-dir data/official
curl -L --fail https://raw.githubusercontent.com/MinaMakar97/MeasuringHateSpeechBigData/c85fd5a3e3eca7c02459f4f229b80ba80cb0ce40/measuring_hate_speech.csv -o data/ratings.csv
curl -L --fail 'https://raw.githubusercontent.com/aequa-tech/ghost-annotator/e69cd23626301a1fcaeba7828fcf350fa580eb30/data/measuring_hatespeech/measuring%20-%20violence.csv' -o data/carrier.csv
curl -L --fail https://raw.githubusercontent.com/CapstoneProject33/Measuring_Hate_Speech/f1e57df64912383bbf239b898a6f956ff119a44c/data/smaller_file.csv -o data/demographics.csv
python attach_research_gender.py --raw data/ratings.csv --carrier data/carrier.csv \
  --evidence data/demographics.csv --output work/ratings_with_gender.parquet
python prepare_population.py --ratings work/ratings_with_gender.parquet --output work/rater_partition.csv
python prepare_split.py --ratings data/official/data/train-00000-of-00001.parquet \
  --rater-partition work/rater_partition.csv --output work/split
```

The official parquet must contain `comment_id`, `annotator_id`, `text` and `hatespeech`. The source checks expect 135,556 ratings and 39,565 distinct texts. `prepare_population.py` retains the original discovery/confirmation seed; `prepare_split.py` then creates the crossed split and anchor assignments. All later scripts read `work/split`.

## Embeddings and ordinal models

```bash
base_revision=49e3418fbbbca6ecbdf9608b4d22e5a407081db4
euro_revision=a25c7fa65fc2a644e6270b8940dbe295b51da681
hf download Qwen/Qwen3-8B-Base --revision "$base_revision" --local-dir "models/$base_revision"
hf download utter-project/EuroLLM-1.7B-Instruct --revision "$euro_revision" --local-dir "models/$euro_revision"
for representation in qwen_base euro; do
  if [ "$representation" = qwen_base ]; then
    model="$base_revision"; model_name=Qwen3-8B-Base
  else
    model="$euro_revision"; model_name=EuroLLM-1.7B-Instruct
  fi
  python extract_embeddings.py --model "models/$model" --input work/split/texts.jsonl \
    --output "work/embeddings_${representation}.npy" --batch-size 8 --max-tokens 1024 \
    --dtype float32 --check-batch-equivalence
  python analyze_thresholds.py --embeddings "work/embeddings_${representation}.npy" \
    --model-name "$model_name" --training-refits 100 --output "work/analysis_${representation}"
  python analyze_calibration_budget.py --embeddings "work/embeddings_${representation}.npy" \
    --analysis "work/analysis_${representation}" --training-refits 100 \
    --output "work/analysis_${representation}_budget"
  python diagnose_calibration.py --analysis "work/analysis_${representation}" \
    --output "work/diagnostic_${representation}_calibration"
  python diagnose_budget_precision.py --embeddings "work/embeddings_${representation}.npy" \
    --analysis "work/analysis_${representation}" --budget-analysis "work/analysis_${representation}_budget" \
    --output "work/diagnostic_${representation}_precision"
done
python analyze_lexical_calibration.py --analysis work/analysis_euro \
  --training-refits 100 --output work/analysis_tfidf_budget
python compare_calibrated_representations.py \
  --embeddings work/embeddings_qwen_base.npy --analysis work/analysis_qwen_base \
  --budget-analysis work/analysis_qwen_base_budget \
  --embeddings work/embeddings_euro.npy --analysis work/analysis_euro \
  --budget-analysis work/analysis_euro_budget --lexical work/analysis_tfidf_budget \
  --training-refits 100 --output work/comparison_representations
for representation in qwen euro tfidf; do
  python analyze_matched_adaptation.py --representation "$representation" --base-directory work \
    --training-refits 100 --output "work/adaptation_$representation"
done
python compare_matched_adaptation.py --analysis work/adaptation_qwen \
  --analysis work/adaptation_euro --analysis work/adaptation_tfidf --output work/comparison_adaptation
```

Keep the directory names in this sequence: the matched analysis loads the saved population parameters, lexical reference and shared draw identities. It selects priors on validation responses before target predictions. Outputs distinguish the original threshold models, the calibration-budget comparisons and the nine matched direction comparisons. The conditional intervals use 2,000 draws; training uncertainty uses 100 refits. A solver or reproduction check failure stops the analysis.
