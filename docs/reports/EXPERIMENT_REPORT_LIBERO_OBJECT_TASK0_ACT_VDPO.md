# LIBERO-Object Task 0 ACT / Variational DPO Pilot Report

## 1. Objective
Integrate one additional dataset source, LIBERO, into the ACT + variational DPO pipeline and run an end-to-end pilot on `libero_object_task0`.

Target task:
- suite: `libero_object`
- task id: `0`
- task name: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`

## 2. Implementation Summary
The repo was extended with:
- a LIBERO task registry and dataset adapter
- direct loading of LIBERO demonstration HDF5s
- padded 14-dim ACT compatibility for 9-dim LIBERO state and 7-dim LIBERO action
- LIBERO closed-loop evaluation in `imitate_episodes.py`
- LIBERO rollout collection compatibility for post-training
- deterministic anchor candidate support for rollout collection
- a separate `libero_act` runtime environment path

## 3. Dataset Format Observed
Downloaded file:
- `/tmp/libero_datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5`

Observed keys inside `data/demo_i/obs`:
- `agentview_rgb`
- `eye_in_hand_rgb`
- `joint_states`
- `gripper_states`
- `ee_pos`
- `ee_ori`
- `ee_states`

Action shape:
- `(T, 7)`

Adapter choice used in this pilot:
- low-dim state = `joint_states (7) + gripper_states (2)`
- padded to ACT's existing 14-dim interface
- action padded from 7 to 14 dims, while runtime env consumes the first 7 dims

## 4. BC Training Pilots
### 1-epoch smoke
Checkpoint dir:
- `/tmp/libero_object_task0_smoke`

Result:
- training pipeline completed successfully
- closed-loop evaluation: **0.0 success** / **0.0 average return**

### 50-epoch BC pilot
Checkpoint dir:
- `/tmp/libero_object_task0_bc50`

Offline training outcome:
- best validation loss: **0.486210** at epoch **47**

Closed-loop temporal-aggregation evaluation result:
- success rate: **0.0**
- average return: **0.0**

Interpretation:
- offline imitation loss improved substantially
- but the policy did not translate to successful closed-loop task completion under the current adapter / hyperparameter setup

## 5. DPO Readiness Probe
Probe rollout root:
- `/tmp/libero_object_task0_rollout_probe`

Settings:
- 2 seed groups
- 2 candidates per seed
- temporal aggregation enabled
- deterministic anchor candidates enabled
- exploratory variants enabled

Probe result:
- total rollouts: **4**
- successes: **0**
- average return: **0.0**

Interpretation:
- there is currently no preference signal for rollout-based DPO on this LIBERO task/checkpoint
- chosen/rejected rollout preferences would collapse because all candidates are equally unsuccessful

## 6. Variational DPO Status
The variational DPO code path is implemented in the repo and validated on transfer cube / insertion tasks, but for LIBERO-Object Task 0 in this pilot:
- the BC checkpoint did not achieve any successful rollouts,
- rollout collection did not produce informative preferences,
- therefore a meaningful LIBERO variational DPO fine-tuning experiment could not be completed yet.

This should be treated as a **blocked-by-baseline** result, not as evidence that variational DPO itself fails on LIBERO.

## 7. Practical Conclusions
1. The LIBERO integration itself is functional:
   - dataset loading works
   - BC training works
   - closed-loop LIBERO evaluation works
   - rollout collection works
2. The current Task 0 ACT baseline remains too weak for rollout-preference DPO.
3. The next work should focus on improving the BC baseline or task representation before retrying DPO.

## 8. Recommended Next Steps
1. Try longer BC training or tuned hyperparameters specific to LIBERO.
2. Try alternative low-dim state choices (e.g. EE-centric state instead of joint+gripper).
3. Add deterministic anchor candidates to a larger rollout collection and re-check whether any success signal appears.
4. Consider a demo-derived preference construction fallback if rollout preferences remain degenerate.
5. Try `window_length=16` once successful rollout preferences exist.

## 9. Key Artifacts
- BC smoke: `/tmp/libero_object_task0_smoke`
- BC 50-epoch pilot: `/tmp/libero_object_task0_bc50`
- rollout probe: `/tmp/libero_object_task0_rollout_probe`
- usage guide: `docs/LIBERO_OBJECT_TASK0_USAGE.md`


## 10. Additional BC Baseline Tuning Attempts (March 11, 2026)
### Fixes applied before retuning
1. Fixed temporal-aggregation occupancy tracking so padded zero action dimensions no longer make valid chunks look unpopulated.
2. Reconstructed the LIBERO runtime environment from demonstration `env_args` metadata instead of using a minimal hard-coded wrapper config.

### Tuning variants attempted
#### A. Joint+gripper state, chunk size 10, 30 epochs
- checkpoint dir: `/tmp/libero_object_task0_chunk10_e30`
- result (10-rollout quick eval): **0.0 success** / **0.0 average return**

#### B. Joint+gripper state, chunk size 10, 100 epochs
- checkpoint dir: `/tmp/libero_object_task0_chunk10_e100`
- best offline val loss: **0.363035** at epoch **81**
- result (10-rollout quick eval): **0.0 success** / **0.0 average return**

#### C. EE+gripper state probe
- added `LIBERO_STATE_SOURCE=ee_gripper` support
- training behavior was similar offline to the joint+gripper variant
- no evidence of early closed-loop success before deprioritizing this branch

#### D. Raw-action training probe
- added `LIBERO_DISABLE_ACTION_NORM=1` support
- checkpoint dir: `/tmp/libero_object_task0_rawact_chunk10_e30`
- best offline val loss: **0.364747** at epoch **22**
- result (10-rollout quick eval): **0.0 success** / **0.0 average return**

### Conclusion from tuning pass
Even after:
- fixing evaluation bugs,
- aligning environment construction with demo metadata,
- reducing chunk horizon,
- increasing BC training length,
- probing alternative state representations,
- probing raw action scale,

closed-loop success on LIBERO-Object Task 0 remained **zero**.

This strongly suggests the remaining problem is no longer a superficial bug. The likely next-level issues are:
1. ACT hyperparameters and observation design may need task-specific redesign for LIBERO.
2. The current 14-dim padding bridge may be too lossy or too indirect.
3. Closed-loop generalization from this demo subset may require a different policy architecture or stronger regularization.
4. Rollout-based DPO remains blocked until BC can achieve nonzero success.
