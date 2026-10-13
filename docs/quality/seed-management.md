# Seed management

This is a reproducibility and review checklist for the Figment simulated Figgie-style market, not financial trading advice.

Keep random seeds explicit in experiment scripts and outputs. Separate training seeds from evaluation seeds. When changing search settings, rerun a fixed held-out seed set and list any exceptions. Do not tune parameters on the final evaluation sample.

## Evidence to retain
- Exact command and source commit
- Random seeds and scenario settings
- Raw outcomes and pass/fail assertions
- Known limitations and follow-up experiments
