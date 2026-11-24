# BME Protocol — Evidence Archive (232 controlled runs)

This repository accompanies the manuscript *Budget-Matched Evaluation for Low-Annotation Industrial Defect Detection: How Effective-Batch Coupling and Learning Rate Confound Reported Gains*. It contains the training and analysis scripts, the split definitions, the per-run configuration files and the final metrics of all 232 controlled runs, so that every quantity reported in the paper can be recomputed from the command line.

The paper is a protocol and benchmark audit. It proposes no new detector, no new architecture and no new loss; it reports how far a conclusion about a structural change can move when the number of optimizer updates is left implicit under a nominally fixed training budget. The central claim is that **fixing the nominal budget is not fixing the actual update count**.

> **Repository scope.** This repository distributes the evidence archive and its browse-friendly indexes. It does **not** contain the manuscript text, the compiled paper, the source images of the two public datasets, or the trained weights. Those exclusions are deliberate and are explained below. The manuscript itself is submitted separately to the journal and is not posted here.

## Repository contents

| Path | Approx. size | Description |
|---|---|---|
| `final_export.zip` | 5.5 MB | Complete evidence set: 6,150 entries, byte-identical to the archive that accompanies the submission |
| `run_index.csv` | 51 KB | **Corrected** index, 232 runs plus header; the 20 Path-A rows carry their per-run `steps_source` annotation |
| `weights_index.csv` | — | All 466 weight files by group, path and size; the weights themselves are not distributed |
| `scripts/`, `root_scripts/` | — | The 22 training, data-preparation, grid and analysis scripts, plus the 6 server-side probe and export scripts |
| `figures/` | ~330 KB | The three manuscript figures as vector PDF plus 300 dpi PNG previews, and `figure_data.json` (per-figure numeric audit table) |
| `patch/pathA_steps.json` | 52 KB | Output of the step-source probe used to correct `run_index.csv` |
| `README.md`, `LICENSE`, `.gitignore` | — | This file, the licence and the exclusion rules |

`final_export.zip` is the smallest complete evidence set. Its md5 checksum is `e16fa873bdef5ae57beee3465c4e3012`.

### Why three archives are not in this repository

| Archive | Size | Reason for exclusion |
|---|---|---|
| `final_dataset.zip` | 101.7 MB | Exceeds GitHub's per-file size limit. Its 44 split yaml files and 5,350 YOLO-format label copies are already inside `final_export.zip` (`data_yaml/`, `data_labels/`); the source images come from the public DeepPCB and NEU-DET datasets. |
| `final_weights_runs_tfa.zip` | 646 MB | Weight files are not needed to recompute any reported number; individual checkpoints can be requested using `weights_index.csv`. |
| `final_weights_runs_b8a.zip` | 828 MB | Same as above. |

The two weight archives are needed only to repeat the validation and IoU-distribution self-checks locally.

## Run accounting: 212 + 20

The 232 runs have two different roles and must not be merged into a single pool.

- 212 runs belong to the main protocol: 182 base grid runs, 5 decisive runs, 15 frozen-backbone capacity-control runs, and 10 step-matched reinforcement runs. These support every result in Sections 5.1 to 5.6.
- 20 runs are the Path-A contextual re-evaluation, in four conditions of five draws each (`u800_b8`, `u3200_b8`, `u800_b30`, `u800_n30`). They support Section 5.7 only.

The 20 Path-A runs are a contextual reference for a published few-shot figure (NEU-DET, 10-shot, mAP50 = 32.8%, reported as the CDT condition of Ghali, Benhafid and Selouani, *Benchmarking Few-Shot Learning Techniques for Steel Surface Defect Detection*, IEEE SWC 2025, DOI 10.1109/SWC65939.2025.00031). The shot definition, the split, the model size and the budget all differ from ours, and the split used for the published figure is not documented. These 20 runs must not be placed in the same table as the published value, must not be described as a reproduction or a like-for-like comparison, and must not be used as an effect estimate against it. The `u800_n30` condition is descriptive only: its split index is not comparable with the 10-shot conditions.

## Directory layout of `final_export.zip`

| Path | Entries | Description |
|---|---|---|
| `final_metrics/`, `args_yaml/`, `results_csv/` | 232 each | Per-run metrics, Ultralytics `args.yaml`, and training curves, under the original run directories |
| `run_index.csv` | 233 lines | One row per run plus header; columns include `group`, `split`, `n_train_images`, `epochs`, `batch`, `batch_requested`, `head_scale_requested`, `freeze`, `mAP50`, `mAP50_95`, `nbs_used`, `accumulate`, `derived_optimizer_steps`, `steps_source` |
| `weights_index.csv` | 467 lines | All 466 weight files with group and size; the weights themselves are not in this archive |
| `root_artifacts.csv`, `manifest.json` | 1 each | Historical artifact list and archive counts, environment and source-formula notes |
| `data_yaml/` | 44 | Split definitions for DeepPCB and NEU-DET at N = 10, 30, 100 and full |
| `data_labels/` | 5,350 | YOLO-format label copies for the sampled splits |
| `scripts/` | 22 | `src/` training, data-preparation, grid and analysis scripts, plus `run.py` |
| `root_scripts/` | 6 | The probe and export scripts actually executed on the server |
| `selfcheck/` | 3 | TFA self-check, stage-2 IoU/confidence distribution, and weight-scan json |
| `vis/` | 16 | Qualitative comparison images from the two self-check passes |
| `docs/` | 9 | Historical how-to files and `requirements.txt` |

The two CSV indexes and the `scripts/` and `root_scripts/` trees are also extracted at the repository root so they can be read directly on the repository page without downloading the archive.

## Environment and derivation

The detector is COCO-pretrained `yolov8s.pt` under `ultralytics` 8.4.208, with `nbs = 64`, `imgsz = 640`, SGD, `seed = 0`, `deterministic = true`, `patience = 1000` and a fixed `last.pt` checkpoint with no model selection. The effective batch is clamped as `batch = min(B, N)`, and gradient accumulation is `accumulate = max(round(nbs / batch), 1)`.

`derived_optimizer_steps` in `run_index.csv` is a value derived from harness accounting, not a step-by-step measurement. The three settings reported in Appendix B of the paper are approximately 100, 400 and 133 updates for `accumulate` values of 8, 2 and 6 respectively. Do not label these as directly counted optimizer steps. The learning-rate arm has no step-matched counterpart at N = 30 and batch = 8, so the step-count mechanism demonstrated in Section 5.3 is not extended to that arm in the paper.

`manifest.json` records one warning: `pathA_steps.json` was not present when the archive was built, so the `steps_source` column of the `run_index.csv` **inside the zip** falls back to `source default nbs=64` for all 232 rows. This repository already applies the correction: the `run_index.csv` at the repository root is the corrected version and takes precedence over the copy inside the zip, and the probe output is preserved in `patch/pathA_steps.json`. The underlying evidence is unaffected: all 20 Path-A `args.yaml` files are present in `args_yaml/` inside the zip.

## Recomputing the reported numbers

All tables and figures in the paper are derived from `final_metrics/` and `run_index.csv` with paired statistics over five stratified draws. The paired design compares matching split indices across conditions; it never subtracts two group means. Reported statistics are paired t-tests with 95% confidence intervals, sign consistency and Cohen's dz.

The three manuscript figures and the per-figure numeric audit table are provided in `figures/` in this repository, not inside `final_export.zip`.

To reproduce, unpack `final_export.zip`, replace the unpacked `run_index.csv` with the corrected copy at the repository root, and use the environment versions listed above. Every reported quantity is a function of `final_metrics/` plus the paired grouping in `run_index.csv`; no weight file, dataset image or network download is required to recompute the tables.

## Interpretation constraints

- `mAP50` and `mAP50-95` can disagree in sign on the same paired weights. The frozen-backbone versus full-fine-tuning comparison at N = 100 is -1.89 on mAP50 and +9.12 on mAP50-95; the latter is a metric-specific observation and must not be written as an overall improvement or extrapolated across IoU thresholds.
- The 1.451 cumulative ratio and the +45% figure refer to box counts accumulated over the four IoU bins from 0.80 to 0.95, not to a single bin.
- The Welch t value of 0.21 in Section 5.1 compares the two final means 85.19 and 85.44 directly; it is not a comparison of the two paired gains.
- The +2.97 gain in Section 5.1 is measured against the five-draw baseline of 82.47, not against the ten-draw baseline of 82.33 in the same table.
- The -3.894 difference in Section 5.7 is marginally non-significant (95% CI [-7.853, +0.065]) and is reported as such.
- Eight early runs in the archive (`g1_full`, `g3_full` and the `shot10_s1` to `shot10_s3` variants of those two groups) carry a `surgery` field with `fca_layers`. They are exploratory snapshots of a frequency-attention candidate that was subsequently rejected; they do not appear in the paper and are not part of any reported result. The `surgery` field in all other runs records only head-width scaling and parameter counts.

## Data availability statement

Suggested wording for the manuscript (append a Zenodo DOI if an archival release is created):

> The training and analysis scripts, the split definitions, the per-run configuration files and the final metrics of all 232 controlled runs are available at `https://github.com/Bittwonotwo/BME-protocol-evidence`. The two public datasets analysed, DeepPCB and NEU-DET, are available from their respective sources.

## Licence

The contents of this repository are released under the **Creative Commons Attribution 4.0 International** licence (CC BY 4.0); see `LICENSE`. The third-party datasets DeepPCB and NEU-DET retain their own terms. If you reuse the archive, cite both the manuscript and this repository.

## Citation

A citation entry with volume, pages and DOI will be added once the article is published. Until then, cite the manuscript by its title and cite this repository as:

> Author(s) (2026). *BME Protocol — Evidence Archive (232 controlled runs)*. GitHub. `https://github.com/Bittwonotwo/BME-protocol-evidence`

