# Pre-existing v45 verifier limitation on the v46/v47 branch

The standalone `verify_v45_freeze.py` reports only `README.md` as mismatched.
This mismatch already exists in parent commit16c86bb: its root README documents
v46, while FREEZE_V45 binds the historical v45 README. The v47 root README is
byte-identical to that parent README. No v45 source or freeze file was changed
or rewritten to make the old verifier pass.

As documented by the existing root README, use historical version tags for those
historical verifiers. The current `verify_v46_freeze.py` passes, including its
preserved historical publication checks. `verify_v47_freeze.py` also passes and
checks that every file in the recorded v46 origin remains unchanged.
