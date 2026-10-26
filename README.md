# TabEXP

LLM-Guided Relational Table Learning via Task-Specific Exploration.

## Installation

Requires Python 3.10 or newer. Install a PyTorch build appropriate for your device,
then install the dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

## Data

Provide a local RelBench dataset directory containing `manifest.yaml`, database
Parquet files, and task splits. Datasets and pretrained models are not downloaded.

## Run

Set `OPENAI_API_KEY` in your environment, then run the stages in order:

```bash
python -m tabexp explore --dataset /data/rel-amazon --task user-churn --output runs/explore

python -m tabexp plan --experience runs/explore/stage1/experience_package.json --output runs/design

python -m tabexp train --plan runs/design/plan.json --program runs/explore/stage1/best_program.json --train-indices runs/explore/evaluation/train_indices.npy --output runs/neural --device cuda
```

Each stage requires a new output directory. Use `--device cpu` for CPU training.
The default model is `gpt-6.1-sol` at `https://api.openai.com/v1`;
use `--model` and `--base-url` on `explore` and `plan` to override them.

The planning defaults are `--max-hops 2` and `--importance-lambda 0.25`.
Training defaults are 10 epochs, patience 2, batch size 512, learning rate 0.001,
and weight decay 0.00001. Models, predictions, plans and metrics are saved under
`--output`. Validation selects the checkpoint; test labels are used only for
final evaluation. Temporary input files are removed after training.

Run `python -m tabexp <explore|plan|train> --help` for all command options.
