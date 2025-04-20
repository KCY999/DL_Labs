@echo off

python dqn.py ^
  --wandb-run-name cartpole-run ^
  --save-dir ./cartpole_results_v2 ^
  --lr 0.0006 ^
  --epsilon-decay 0.9998 ^
  --target-update-frequency 300 ^
  --replay-start-size 1000 ^
  --max-episode-steps 500


