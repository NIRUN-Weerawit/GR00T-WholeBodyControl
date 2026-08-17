# G1 Tennis Swing Primitive and SONIC-Token Policy Plan

> **Scope:** This plan covers a single safe, repeatable racket-swing primitive on the Unitree G1 in simulation first. It does **not** attempt a tennis rally, ball return, serve, robot deployment, or full tennis VLA.

**Goal:** Record human tennis swing primitives with PICO, convert them into physically valid G1 reference motions, execute them with frozen SONIC, and later learn autonomous SONIC-token generation without replacing SONIC's low-level controller.

**Architecture:** Keep SONIC frozen as the whole-body execution layer. A primitive player first replays validated motion-token trajectories. A later imitation-learning policy takes task conditions and predicts valid SONIC motion tokens; SONIC's decoder combines those tokens with live G1 proprioception to generate 29-D low-level actions.

**Core principle:** The learned task policy decides *what motion should happen*. SONIC decides *how G1 executes that motion while maintaining coordinated whole-body control*.

---

## 1. Final System Boundary

```text
                 HIGH-LEVEL POLICY (later)
text / vision / ball state / robot state / phase
                    |
                    v
          SONIC motion-token trajectory
                    |
                    v
      frozen SONIC decoder + G1 proprioception
                    |
                    v
          29-D low-level G1 body action
                    |
                    v
                 PD / motors
```

The high-level policy must **not** initially output direct G1 joint positions. Direct joint control would require it to rediscover balance, contact robustness, smooth transitions, and whole-body coordination that SONIC already provides.

SONIC's supported VLA action interface is:

```text
64-D motion token
+ 7-D left hand command
+ 7-D right hand command
= 78-D action
```

For the right-handed swing prototype, the important command is the 64-D body-motion token. Gripper/hand behavior can initially remain fixed solely to hold the racket safely.

---

## 2. Near-Term Deliverable

Demonstrate in MuJoCo:

```text
stable ready stance
→ wind-up
→ right-handed forehand-like swing
→ follow-through
→ stable recovery stance
```

The first demonstration is autonomous because a trigger invokes a deterministic primitive state machine. It is **not** yet ball reactive.

### Acceptance criteria

| Criterion | Initial requirement |
|---|---|
| Execution | Completes the whole sequence autonomously from a trigger |
| Stability | No fall or simulator termination across repeated executions |
| Safety | No joint-limit violation, self-collision, or sustained foot penetration |
| Motion quality | Clear body rotation, arm swing, follow-through, and recovery visible in video |
| Repeatability | Can execute several swings from the recovered ready pose |
| Racket proxy | A smooth, repeatable racket trajectory; no ball contact requirement yet |

---

## 3. What to Record

### 3.1 First primitive only

Record one narrow, deliberately repeatable human motion family:

```text
right-handed stationary forehand
```

Each take must follow the same semantic structure:

```text
ready stance
→ unit turn / wind-up
→ forward swing
→ follow-through
→ recovery to ready stance
```

Keep these fixed for the first dataset:

- dominant hand;
- grip style;
- stance family;
- intended contact height;
- direction of swing;
- start and end pose;
- court-space location.

Do not initially mix forehands, backhands, serves, volleys, lateral running, and full rallies. Those are distinct primitives and should become separate labelled datasets later.

### 3.2 Dataset quantity

Start with a pilot, not a large collection:

```text
20–50 controlled takes of one mid-height stationary forehand.
```

Use the pilot to validate PICO tracking, racket calibration, temporal filtering, G1 retargeting, and frozen-SONIC execution. Expand data only after the full pipeline produces a stable G1 swing.

---

## 4. PICO Recording Design

### 4.1 Run as a pure human-motion recorder

The recorder must run without:

```text
MuJoCo
G1 deployment binary
camera server
Remote Vision bridge
PICO video feedback loop
run_data_exporter.py
```

This removes teleoperation latency from human movement. PICO is used only as a wearable motion-capture system.

### 4.2 Preserve raw source data and derived SONIC-compatible fields

For every source frame, preserve:

```text
PICO/XRoboToolkit timestamp
24 body-joint poses                (24, 7): xyz + xyzw quaternion
24 body-joint velocities           (24, 6), if available
24 body-joint accelerations        (24, 6), if available
headset pose                       (7,)
left/right controller poses        (7,)
left/right raw ankle-tracker poses (7,)
left/right tracker velocity/acceleration, if available
```

Also save derived data in the repository's SONIC convention:

```text
smpl_pose        (63,)      # 21 local axis-angle rotations
smpl_joints      (24, 3)    # root-local, Z-up joint positions
body_quat_w      (4,)       # root orientation in SONIC convention
```

The existing `pico_manager_thread_server.py` receives raw 24-joint pose data but currently publishes/saves primarily derived, interpolated stream fields. The new recorder must preserve the raw signals separately so coordinate conventions, calibration, filtering, and retargeting can be improved later without re-recording the athlete.

### 4.3 Racket representation is mandatory metadata

SMPL wrist motion is not an accurate substitute for racket pose. Every take must include a racket 6D trajectory obtained by one of the following methods:

| Priority | Method | Requirement |
|---:|---|---|
| Preferred | Dedicated PICO motion tracker fixed to the racket | Track it in object-tracking mode and record its 6D pose |
| Minimum | Right controller held rigidly with racket | Calibrate and save a fixed `T_racket_from_controller` transform |
| Not acceptable for later ball contact | Infer racket pose solely from body wrist joint | May be used for body-only visualization but not racket-face evaluation |

The calibration transform must be stored per recording session:

\[
T_{racket}^{world} = T_{controller}^{world} T_{racket\leftarrow controller}
\]

### 4.4 Episode recorder for state-machine replay (lerobot-protocol preserving)

A dedicated, deploy-independent recorder supplies the primitive player (Task 7)
with per-episode replay inputs. It is **not** `run_data_exporter.py` (that saves
the full lerobot dataset from robot + SMPL + camera); it records only the essential
fields a replay state machine needs, **reusing the exporter's protocol unchanged**:

- **Wire protocol identical to `run_data_exporter.py`:** ZMQ SUB on `pose`,
  `planner`, `manager_state` at `:5556`; frames unpacked with `unpack_pose_message`
  (1280-byte JSON header + concatenated little-endian fields).
- **Episode state machine identical:** `EpisodeState` RECORDING → NEED_TO_SAVE →
  IDLE, plus abort/discard.
- **Episode controls reused:** the existing `toggle_data_collection` /
  `toggle_data_abort` manager flags — **Left-Grip+A** start/stop-save,
  **Left-Grip+B** discard — consumed from `manager_state`.

Output difference: instead of the lerobot parquet + MP4 physical layout, it writes
**one self-contained `.npz` per episode**, labeled with the tennis primitive:

```text
episodes/<primitive>_<clip_id>.npz
    smpl_pose           (T, 21, 3)
    smpl_joints         (T, 24, 3)
    body_quat_w         (T, 4)
    left_wrist_joints   (T, 3)
    right_wrist_joints  (T, 3)
    vr_3pt_position     (T,)    vr_3pt_orientation (T,)
    frame_index         (T,)
    timestamp           (T,)           # original PICO timestamps preserved
    primitive_label     str            # handedness, stance, contact height,
                                       # session/calibration id
```

This no-deploy recorder must run without MuJoCo, deployment binary, camera server,
or Remote Vision (per 4.1). The saved episode feeds the primitive state machine
directly: the phase-indexed trajectory becomes the reference window for
reference-window or cached-token replay (Task 7).

---

## 5. Offline Processing and Retargeting

### 5.1 Clean and segment

For each recorded take:

1. Reject corrupt tracking segments, missing samples, discontinuous quaternions, or dropped-frame bursts.
2. Resample after cleanup to SONIC's 50 Hz control timeline while preserving original timestamps.
3. Use quaternion interpolation for orientations and derive velocities after resampling.
4. Segment the take with explicit boundaries:
   ```text
   ready / wind-up / swing / follow-through / recovery
   ```
5. Attach labels: primitive name, handedness, stance, intended contact height, take ID, capture-session calibration ID.

### 5.2 Convert to a paired SONIC training/reference pair

A human SMPL sequence alone is insufficient for SONIC motion tracking training. Each accepted human recording must be paired with its matching G1 retargeted motion:

```text
PICO body + racket trajectory
→ human-motion cleanup
→ G1 kinematic retargeting
→ G1 feasibility filtering / refinement
→ paired files with identical clip ID
```

The pair contains:

```text
smpl/<clip_id>.pkl
    pose_aa
    smpl_joints
    transl
    fps

g1_retargeted/<clip_id>.pkl
    29-D G1 joint motion and required body kinematics
    fps
```

The G1 trajectory is the physical reference. The SMPL trajectory is the human-motion command/reference modality. They must have matching frame counts after resampling.

### 5.3 Validate before any learning

Visualize in simulation:

```text
PICO skeleton
→ retargeted G1 kinematic motion
→ frozen SONIC simulated execution
```

Reject or repair a motion if it creates:

- impossible G1 joint ranges;
- severe foot sliding;
- unstable centre-of-mass motion;
- self-collision;
- impossible torso twist;
- unachievable racket path;
- failure to recover to a controllable pose.

---

## 6. Phase 1: Frozen-SONIC Primitive Replay

### Objective

Validate that frozen SONIC can execute the retargeted G1 forehand reference robustly. Do not fine-tune SONIC before this test.

### Execution options

#### Option A — Reference-window replay

```text
primitive state machine
→ select phase-indexed G1 reference window
→ frozen SONIC encoder
→ 64-D token
→ frozen SONIC decoder
→ G1 action
```

This is the preferred first path because it exercises the released supported G1-reference interface.

#### Option B — Cached token replay

After Option A passes, cache the token produced at each phase:

```text
validated G1 reference window
→ frozen SONIC encoder
→ token z*(phase)
```

Then execute:

```text
primitive state machine
→ replay cached z*(phase)
→ frozen SONIC decoder + current G1 state
→ G1 action
```

This has the same interface expected by a later VLA/IL policy.

### Primitive state machine

```text
IDLE
  → READY
  → WIND_UP
  → SWING
  → FOLLOW_THROUGH
  → RECOVER
  → READY
```

The initial transition trigger may be keyboard-only. A visual or ball-conditioned trigger is deliberately deferred.

### Decision gate

- **If frozen SONIC tracks the motion stably:** retain frozen SONIC.
- **If frozen SONIC fails despite a physically plausible retargeted trajectory:** investigate the tracking failure. Only then consider a narrow SONIC fine-tune using paired G1 + SMPL tennis clips.

Fine-tuning is a corrective measure, not the default plan.

---

## 7. Phase 2: Imitation Learning in SONIC Token Space

### Objective

Replace deterministic token replay with a learned policy that produces valid SONIC token trajectories conditioned on the task and current state.

### Teacher targets

Do not train against arbitrary 64-D vectors. Build IL targets by running each validated primitive through frozen SONIC's encoder:

\[
z_t^* = E_{SONIC}(\text{reference window at } t)
\]

The IL policy should imitate these valid teacher token trajectories:

\[
\hat z_{t:t+H} = \pi_{IL}(o_t, c_t)
\]

where:

- \(o_t\) is current G1 proprioception;
- \(c_t\) is task conditioning;
- \(H\) is the output token horizon;
- \(z_t^*\) is a frozen-SONIC teacher token.

A first objective is token regression plus temporal consistency:

\[
\mathcal{L}_{token} = \|\hat z_t - z_t^*\|^2
\]

\[
\mathcal{L}_{smooth} = \| (\hat z_t-\hat z_{t-1}) - (z_t^*-z_{t-1}^*) \|^2
\]

### Conditioning curriculum

| Version | Inputs | Output | Purpose |
|---|---|---|---|
| 0 | Trigger only | Fixed cached token sequence | Prove autonomous replay |
| 1 | Primitive label / text | Primitive token sequence | Select among a small library |
| 2 | G1 proprioception + primitive label + phase | Next token/chunk | Adapt primitive to small state errors |
| 3 | Ball state + G1 state + desired landing target | Token chunk + phase rate | Select and time a stroke |
| 4 | Vision + G1 state + task text | Token chunk | End-to-end perception-conditioned behavior |

Text is appropriate for selecting a primitive (for example, “perform a forehand”). It is not sufficient for real-time tennis timing. Ball-reactive behavior needs measured or estimated ball position, velocity, and predicted time-to-contact.

---

## 8. Phase 3: RL Residuals, Not Full Joint-Space RL

Only add RL after stable primitive replay and/or token-space IL are verified.

### Initial RL targets

Use RL for robot-specific improvements unavailable in human PICO motion:

```text
stronger but bounded swing velocity
whole-body stability under fast torso rotation
stable recovery after follow-through
racket-face orientation correction
later: racket-ball contact and return direction
```

Keep the learned correction bounded around the primitive/SONIC behavior:

\[
a_t = a_t^{SONIC} + \Delta a_t^{RL}
\]

Do not begin with unrestricted 29-joint RL actions. Start with limited residual authority and increase only after measurement proves it is safe.

For actual ball contact, preserve an explicit high-bandwidth right-wrist/racket correction pathway. Whole-body primitive style can come from SONIC; precise racket-face orientation and contact timing require more specialized control.

---

## 9. Explicit Non-Goals for the First Milestone

The following are outside the first implementation and must not expand scope:

```text
full rallying
ball perception
ball-flight prediction
serve and overhead motions
court-scale locomotion
robot hardware deployment
competitive return placement
full VLA training
end-to-end vision-to-joints training
```

---

## 10. Implementation Order

### Task 1: Define the capture schema

**Objective:** Specify one raw per-take file format and calibration manifest.

**Expected artifacts:**

```text
raw/<clip_id>.npz
metadata/<clip_id>.json
calibration/<session_id>.json
```

**Verification:** A synthetic take can be written and read back with all required arrays, shapes, timestamps, and calibration identifiers intact.

---

### Task 2: Implement the standalone PICO recorder

**Objective:** Read PICO/XRoboToolkit full-body data and controller/tracker data without launching simulator or robot processes.

**Expected behavior:**

```text
record start
→ record one raw human take
→ record stop
→ one self-contained raw file plus metadata
```

**Verification:** Collect a 10-second stationary calibration recording; verify monotonic timestamps, 24 body joints per frame, and no NaNs.

---

### Task 3: Implement racket calibration and recording

**Objective:** Produce a world-frame racket trajectory from a dedicated tracker or controller-to-racket transform.

**Verification:** Hold racket still, rotate it around each axis, and confirm plotted racket orientation follows continuously with no axis/sign inversion.

---

### Task 4: Implement offline cleanup and 50 Hz export

**Objective:** Convert raw PICO takes into cleaned SONIC-convention SMPL trajectories while retaining raw originals.

**Verification:** A known clip exports exactly 50 frames per second with continuous unit quaternions and matching data lengths.

---

### Task 5: Retarget one forehand to G1

**Objective:** Produce one paired `smpl/<clip_id>.pkl` and `g1_retargeted/<clip_id>.pkl` sequence.

**Verification:** Visualize human and G1 skeletons together; check joint bounds, foot-ground clearance, and valid recovery pose.

---

### Task 6: Validate frozen SONIC reference execution in MuJoCo

**Objective:** Run the retargeted motion through frozen SONIC repeatedly.

**Verification:** Save one combined execution video and report termination count, joint-limit violations, and final recovery stability over repeated runs.

---

### Task 7: Implement the deterministic primitive player

**Objective:** Trigger ready → swing → recovery through reference-window or cached-token playback.

**Verification:** Trigger five consecutive executions in simulation. Each must return to the ready state without manual reset.

---

### Task 8: Extract and validate teacher token sequences

**Objective:** Cache the actual 64-D SONIC token trajectory corresponding to each validated primitive.

**Verification:** Compare reference-window execution against cached-token execution. Confirm both remain stable and produce qualitatively matching swings.

---

### Task 9: Train a small phase-conditioned token IL policy

**Objective:** Predict teacher token chunks from G1 state, primitive label, and phase.

**Verification:** On held-out starts near the ready pose, compare token error, temporal smoothness, swing completion rate, and stability against deterministic replay.

---

### Task 10: Decide whether SONIC fine-tuning is warranted

**Objective:** Make a measurement-based decision rather than assuming fine-tuning is needed.

**Fine-tune only if all are true:**

1. G1 retargeted motion is physically plausible;
2. deterministic reference playback fails under frozen SONIC;
3. the failure is reproducible and diagnosed as tracking capacity/domain mismatch, not a data, calibration, or retargeting bug.

---

## 11. Success Ladder

```text
L0  Raw PICO + racket data recorded correctly.
L1  One human forehand retargets to a valid G1 reference.
L2  Frozen SONIC executes one stable simulated forehand.
L3  State machine autonomously replays the primitive repeatedly.
L4  Token-space IL reproduces primitive under small state variation.
L5  RL residual improves safe power and recovery.
L6  Ball-conditioned policy selects/times primitives.
L7  Racket-ball contact is simulated and evaluated.
```

No level should be started until the previous level has measurable evidence of success.

---

## 12. References

- `gear_sonic/scripts/pico_manager_thread_server.py` — current PICO body-pose conversion and SONIC stream publisher.
- `gear_sonic/utils/teleop/input_readers.py` — current PICO/XRoboToolkit reader.
- `gear_sonic/scripts/run_vla_inference.py` — external motion-token publisher.
- `docs/source/tutorials/vla_workflow.md` — SONIC token action interface.
- `docs/source/tutorials/vla_inference.md` — 78-D SONIC embodiment action space and deployment flow.
- `docs/source/user_guide/training.md` — paired G1/SMPL SONIC training layout.
- `docs/source/references/motion_reference.md` — G1 reference-motion requirements.
- LATENT: https://arxiv.org/html/2603.12686
- SONIC: https://arxiv.org/html/2511.07820
