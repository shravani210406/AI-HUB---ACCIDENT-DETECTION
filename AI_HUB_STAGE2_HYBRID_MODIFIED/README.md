# AI Hub Stage 2 — Hybrid Modified

This is the Stage 2 project layout with its default model logic replaced by the
Hybrid inference system: YOLO11n road-object detection, ByteTrack tracking,
motion/interaction features, CLIP scene features, a trained temporal TCN, and
calibrated K-of-N / hysteresis event decisions.

The original Stage 2 folders and entry point are retained. `main.py` keeps the
same experiment-file workflow: it reads `experiment_videos`, finds every
listed video anywhere under the project folder, processes them one by one, writes
JSON logs, and creates the same Excel/CSV reports. Hybrid technical
diagnostics are saved separately under each experiment's `hybrid_diagnostics/`
folder, so they cannot interfere with the original post-processing reports.

## Run

Put your videos in any folder inside this project, for example:

```text
videos/
  acc54.mp4
  norm33.mp4
```

Then run:

```powershell
python main.py
```

Each line in `experiment_videos` must be `filename,label`, for example
`acc54.mp4,accident` or `norm33.mp4,normal`. You may instead use a full video
path in the first column. The original `new_dataset_final/new_acc` and
`new_dataset_final/new_norm` folders also continue to work.

The required Hybrid detector, CLIP, temporal model, and OpenVINO artifacts are
included in `models/`. The active calibrated settings are in
`configs/default.yaml`. Historical comparisons, benchmark runs, training scripts,
cached features, and past reports are deliberately not included or called by the
default program.
