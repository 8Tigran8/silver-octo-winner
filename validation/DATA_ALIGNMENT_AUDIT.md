# Data lineage and alignment audit

Official commit `f8f2629ff1e0985f81d7e59d8413678514ee1530` was retrieved from the current remote main on 16 November. See `official_sources.json`; the complete raw specification is retained locally under `reports/independent_audit/official`.

| Stage | Contract / time base | Verified behaviour and limitations |
|---|---|---|
| XML / raw ATD | Two teams, S slot mappings; N consecutive frame IDs 1..N | First team is home, as official baseline specifies. Unknown team references now fail. Parser row i has ID i+1; load rejects missing, duplicate or out-of-order IDs rather than silently joining them. |
| Tactical alignment | N=199319, 25Hz, normalized xy | Generator now explicitly verifies N. Synthetic first/last mapping agrees with frame_id-1. No broadcast index is used. |
| Raw arrays | players N×S×3; ball N×3 | Missing observations remain NaN. Speeds are m/s; normalized xy uses pitch dimensions in metres. Slots describe detections, not persistent identities. |
| Base features | N×87, plus direct/team-swapped view | Previously interpolation, derivatives and smoothing crossed period boundaries. Repaired by computing every base feature independently inside each declared period; outside rows remain NaN. Tiny windows now preserve input length. |
| Geometry | pitch default 105.3×68; external pitch override | Known synthetic nearest distances verified in metres. Equal-distance nearest-speed ties now average finite tied observations; no slot-order tie winner. Duplicate observations can still alter counts/centres: that changes observed data and is a robustness risk. |
| Contact context | N×108 | Forward/backward observed-contact carries and rolling averages reset per period. |
| Base / stack / team | per-frame dead probability and home probability conditional on alive | Stacking previously pooled across boundaries; inference now passes periods explicitly. Unit/full-predictor tests change the other period and require exact equality in the first. |
| Neural inference | 5Hz, 108 features plus finite masks | Global stride-five sample phase preserved; half-open downsample spans use ceil(start/5), ceil(stop/5). Context clips to each period. Linear upsampling resets per period. Global sample phase is not invariant to arbitrary non-multiple-of-five temporal shifts; no such false invariance is claimed. |
| Decoder | four internal states H/A/dead-after-H/dead-after-A | Cost-9 duration Viterbi; five 5Hz steps for a minimum one-second state duration. Periods decoded separately. D states both output D. |
| Final output | 199319 ordered rows frame,label | X outside official blocks; published second kickoff is retained. Official format validation and two clean-environment reproductions pass. |

Published blocks are [11185,90652) and [110808,190198), 158857 graded frames. The second kickoff constant 113742 is explicitly present in official `starter/load_smart_tagging.py` and `starter/load_atd.py`; it is not inferred from private or public-window labels. The initial second-half pre-kickoff interval is D. Raw Smart Tagging, homography and broadcast are not read by this inference path.

NaN / missingness: no-ball, one-team-dropout and all-player-missing synthetic inputs yield finite probabilities and deterministic predictions. These demonstrate defined execution, not correctness of guessed ownership. Complete input validation of every possible corrupt numeric token/Inf or nonsensical coordinate is not established. Out-of-pitch coordinates are retained as meaningful geometry. Reference labels are rejected as predictor inputs.

The pre-fix failure receipts are preserved. Correctness repairs do change feature/probability semantics; old validation metrics must not automatically be attributed to this code. On the supplied match, final CSV bytes happen to remain identical, confirmed without opening target labels.
