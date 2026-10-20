# Reproduce

## Data + tokenizer (once)
    uv sync
    uv run python data/fetch.py
    uv run python tokenizer/bpe.py train --input data/sample.txt --out tokenizer/bpe16k.json
    uv run python data/prepare.py --tok tokenizer/bpe16k.json --input data/raw/train.txt --output data/train.npy
Gives train.npy ~140M toks, valid.npy 5.26M toks.

## Base 49.9M (8x1)
    uv run python training/train.py --data data/train.npy --valid data/valid.npy --vocab 16000 --d 512 --layers 8 --heads 8 --ff 2048 --ctx 512 --steps 12000 --batch 48 --lr 6e-4 --warmup 400 --ckpt checkpoints/s60
    uv run python eval/eval.py --tok tokenizer/bpe16k.json --ckpt checkpoints/s60

## Looped 4x2, 33.2M (model/gpt_looped.py, training/train_looped.py)
    uv run python training/train_looped.py --data data/train.npy --valid data/valid.npy --distinct 4 --passes 2 --steps 1000 --batch 48 --lr 6e-4 --ckpt checkpoints/looped-4x2-pilot --seed 0
    # copy pilot ckpt to full dir, then:
    uv run python training/train_looped.py --data data/train.npy --valid data/valid.npy --distinct 4 --passes 2 --steps 12000 --batch 48 --lr 6e-4 --ckpt checkpoints/looped-4x2 --resume --seed 0
    uv run python eval/eval_looped.py --tok tokenizer/bpe16k.json --ckpt checkpoints/looped-4x2 --distinct 4 --passes 2

## Looped 2x4, 24.8M
Same with --distinct 2 --passes 4, ckpts checkpoints/looped-2x4-pilot and checkpoints/looped-2x4.

## Config
AdamW wd 0.1, lr 6e-4, warmup 400 linear, cosine to 0.1x, batch 48, ctx 512, seed 0. Log every 25, ckpt+valid every 500. Full write-up: RESULTS-looped-3way.md in the source repo.
