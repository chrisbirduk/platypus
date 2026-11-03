# Colab

`platypus_gpu.ipynb` runs one build on a Colab GPU:

1. Set the build, preset, seed and any control in the first cell.
2. Run all. The notebook mounts Drive, clones the repository (or uses an
   uploaded build file), runs `scripts/run_gpu.sh`, and prints the
   readout summary.
3. The run folder (log, manifest, readouts) is saved under
   `MyDrive/platypus/runs/`, and the streamed Pile text is cached under
   `MyDrive/platypus/pile_cache/` so later runs start faster.

A full three-pass run takes four to five hours on one large GPU (95 GB
class). Colab sessions can end before that; the run folder on Drive keeps
everything printed up to that point.

Secrets: keys stay in Colab Secrets. The notebook reads them with
`google.colab.userdata.get` and never prints them.
