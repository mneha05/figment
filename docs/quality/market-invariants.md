# Market invariants

This is a reproducibility and review checklist for the Figment simulated Figgie-style market, not financial trading advice.

Every fill must conserve the 40 cards across all players and preserve total cash after accounting for the game pot. Check that agents cannot offer more cards than inventory or bid beyond available cash. Assert price-time priority with simultaneous quoting cases.

## Evidence to retain
- Exact command and source commit
- Random seeds and scenario settings
- Raw outcomes and pass/fail assertions
- Known limitations and follow-up experiments
