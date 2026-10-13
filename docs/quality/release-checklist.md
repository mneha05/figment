# Release checklist

This is a reproducibility and review checklist for the Figment simulated Figgie-style market, not financial trading advice.

Run the documented tests and deterministic example before tagging a version. Review changed payout and ordering logic for regressions. Include the selected seed, command, Python version, and produced artifacts in any release note. Never claim production trading suitability.

## Evidence to retain
- Exact command and source commit
- Random seeds and scenario settings
- Raw outcomes and pass/fail assertions
- Known limitations and follow-up experiments
