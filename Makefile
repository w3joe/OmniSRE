.PHONY: setup test lint gen-dataset eval sft rl-tier1 rl-tier2 serve

PYTHON ?= uv run python

setup:
	uv sync --extra dev

test:
	uv run pytest -v

lint:
	uv run ruff check src tests scripts

gen-dataset:
	$(PYTHON) scripts/build_datasets.py \
		--output-dir data/generated \
		--num-scenarios 600 \
		--sft-trajectories 400 \
		--seed 42

eval:
	$(PYTHON) scripts/run_eval_pass_hat_k.py \
		--scenarios data/generated/eval_heldout_scenarios.json \
		--policy oracle_noisy \
		--k 5

sft:
	bash scripts/run_sft_4xh100.sh

rl-tier1:
	bash scripts/run_rl_tier1_4xh100.sh

rl-tier2:
	bash scripts/run_rl_tier2_live_4xh100.sh

serve:
	bash scripts/serve_flash_sglang.sh
