# Tools

This public repository includes only the benchmark-facing tool subset.

- `ccbench.py`: compact public entry point for manifests, CCP referee runs,
  native ladder samples, runner spot checks, and Elo estimation.
- `ccp_referee.py`: trusted stdin/stdout CCP match runner.
- `ccp_cczero_bot.py`, `ccp_cczero_engine.py`: adapters for built-in bots and
  `.ccpv` model checkpoints.
- `run_baseline_ladder.py`, `run_baseline_ladder_native.py`: baseline ladder
  runners.
- `run_champion_match.py`, `sprt_decision.py`: direct champion-match and SPRT
  helpers.
- `audit_position_suite.py`, `audit_position_suite_native.py`,
  `freeze_position_suite.py`, `build_position_suite.py`: position-suite
  auditing and freezing helpers.
- `estimate_elo.py`: Bradley-Terry Elo estimator for benchmark JSON/JSONL.
- `validate_jsonl.py`, `validate_runner_spotchecks.py`: validation helpers.
