# Superseded broader v46 candidate

Original published commit: `8f12c202944b0a01d3c6898991612b04b117eab6`.
Source and prior CPU/pilot evidence remain at
`stage21_v46_controlled_ac_20261104_v1/` and `evidence/v46/`.
`FREEZE_V46_original.json` preserves its original release source hashes.
The root README and verifier are archived as `ROOT_README_original.md` and
`verify_v46_freeze_original.py`; all other frozen paths retain their old bytes.
The current verifier checks all57 original frozen file contents through this map.

The planned 18-arm matrix stopped after three pure200 GPU replays completed:
original A/GPU0 attained7/200, the broader v46/GPU1 attained30/200, and A*/GPU2
attained12/200. Each returned all200 chains. The offline gate divided by zero
for empty foreground request classes, so no within-GPU arm triplet completed
and no formal mixed arm started. These three observations are not a complete
paired comparison. Original failure statuses/logs and full raw archives remain
preserved in the research workspace and server study.

Historical v45 full configurations attained43--45/200. The broader candidate
showed a regression signal and more preemption; this motivated the user's
explicit request to rebuild v46 from v45 with only mixed coexistence and
remaining-work boundary fixes. The replacement is under
`stage22_v46_v45_minimal_20261105_v1/`. No history was rewritten and no old result
has been reassigned to the revised version. The replacement has no GPU result.
