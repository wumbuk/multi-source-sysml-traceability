# Experiment Results

This directory retains compact prediction outputs, metrics, mismatch samples,
and ablation artifacts used in the study.

The repository intentionally excludes:

- raw SFT/chain-of-thought generation logs;
- temporary recovery runs;
- downloaded model weights;
- duplicated build or cache files.

For each reported run, retain enough metadata to identify:

- model name and exact revision;
- LLM server and library versions;
- decoding parameters and reasoning mode;
- embedding model revision;
- dataset/gold-label revision;
- hardware and driver versions;
- random seeds;
- RQ1 and RQ2 metric JSON files.

Do not commit API responses that expose credentials, private prompts, or
provider-restricted reasoning traces.
