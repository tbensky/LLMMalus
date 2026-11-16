# Running Malus against Ollama models on CSU TIDE

The same setup as GAPhysicsSolver's `k8s/`: an image with this repo in it, a
GPU Job with an Ollama sidecar, and a claim that caches downloaded models so
each one is pulled once. `./malusctl` (in the repo root) does the kubectl work.

You need what TIDE needs: an NRP account, membership in `cal-poly-physai`,
`kubectl` pointed at the cluster, and the `ghcr-creds` pull secret that
GAPhysicsSolver already uses in that namespace. Building the image needs
`docker buildx` and a login to ghcr.io (`docker login ghcr.io`).

## Once

```bash
./malusctl build      # build the image and push ghcr.io/tbensky/llmmalus:latest
./malusctl volumes    # create malus-models (100Gi) and malus-results (10Gi)
```

Rebuild with `./malusctl build` after changing any Python or prompt: the Job
runs whatever the image holds, not your working copy. The image records its
git commit, and every run's `summary.json` has it as `code_version` (with
`-dirty` if there were uncommitted changes at build time).

The first push creates the `llmmalus` package on ghcr.io as private. If the
Job's lab container then fails with `ImagePullBackOff`, either give the
package access to the account behind `ghcr-creds`, or make it public under
the package's settings on GitHub.

## Models

```bash
./malusctl models                       # what is cached, disk left, partial downloads
./malusctl pull qwen3:8b gemma3:27b     # download ahead of time (no GPU)
./malusctl rm-model gemma3:27b          # remove one (refuses while a batch runs)
```

These run as small Jobs with no GPU, so they schedule at once. Pulling ahead
is optional: a batch's own init container pulls its model first if the cache
lacks it. Check `models` before a big pull; a 27-32B model at Q4 is ~16-20 GB.

## A batch

```bash
./malusctl start qwen3:8b 10                         # 10 runs, one after another
./malusctl start gemma3:27b 5 -- --keep-last 10      # anything after -- goes to run_lab.py
./malusctl pull-status     # follow the model download (first batch for a model)
./malusctl status          # results so far, read off the claim
./malusctl log             # the lab's output as it happens
./malusctl stop            # delete the Job
```

`status` is the one to use while a batch runs. It reads each run's
`summary.json` (or, for the run in progress, its transcript) straight off the
claim in one shot, so the laptop sleeping does not matter. `log` follows
kubectl's stream, which dies when the laptop sleeps; it starts from the last
2000 lines (`MALUS_LOG_TAIL`) so restarting it is cheap.

Every run is a fresh conversation; Ollama keeps the model loaded on the GPU
between runs. Runs go to `Runs/<model>/NNN` on the results claim, numbered
across batches exactly as locally, with `jobs.log` beside them recording each
batch's pod, node, image version and settings.

### Settings

| Environment | Default | |
| --- | --- | --- |
| `MALUS_NUM_CTX` | 32768 | Ollama's context window per request. Past it Ollama silently drops the oldest messages, task included; for long runs of a wordy model use `-- --keep-last N` instead of an ever larger window |
| `MALUS_TIMEOUT` | 600 | seconds per model call |
| `MALUS_GPU` | NVIDIA-L40 | 46 GB, 17 nodes. `NVIDIA-A100-80GB-PCIe` (80 GB, 3 nodes) for a model that does not fit; record when you switch, since the card is a variable |
| `MALUS_IMAGE` | ghcr.io/tbensky/llmmalus:latest | |
| `MALUS_NAMESPACE` | cal-poly-physai | |

For example `MALUS_NUM_CTX=65536 ./malusctl start qwen3:32b 5`. The Ollama
container has 24Gi of host memory; a model much over ~20 GB may need more
(`k8s/job-gpu.yaml`), but raise it only after the pod reports OOMKilled.

Thinking models: `-- --think off` (or `on`) sets Ollama's thinking switch;
leaving it out uses the model's default. A thinking model can spend its whole
context window thinking, which shows up as empty replies and
`no_tool_calls`.

## Getting results

```bash
./malusctl fetch qwen3:8b    # one model
./malusctl fetch             # every model on the claim
```

Runs land in `Runs/k8s/<model>/NNN`, apart from local runs (which also
number from 001). Each run is copied as its own verified archive, and a run
already fetched with its `summary.json` is skipped, so `fetch` can be re-run
until it reports no failures, including while a batch is still going (the
run in progress is fetched again next time). It works through a small
`malus-pvc-shell` pod that it creates and deletes; delete it yourself
(`kubectl delete -n cal-poly-physai pod/malus-pvc-shell`) if a fetch is
interrupted.

## Files

```text
malusctl               the commands above
k8s/Dockerfile         the lab image (python:3.12-slim + matplotlib + this repo)
k8s/entrypoint.sh      lab container: wait for Ollama, run_lab.py --runs N, release the sidecar
k8s/job-gpu.yaml       a batch: ollama-pull init, lab, Ollama on one GPU
k8s/models-job.yaml    list / pull / remove models, no GPU
k8s/volumes.yaml       malus-models and malus-results claims
k8s/pvc-shell.yaml     a pod on the results claim, for fetch
```

`job-gpu.yaml` and `models-job.yaml` have placeholders that `malusctl` fills
in, so create them through `malusctl` rather than `kubectl create -f`.

Differences from GAPhysicsSolver's setup, on purpose:

- The lab tells the Ollama sidecar to stop through a file on a pod-local
  `emptyDir`, not on the shared results claim, so two batches on the claim at
  once cannot delete each other's stop marker.
- The claims are Malus's own (`malus-models`, `malus-results`), so pruning or
  filling one project's cache cannot touch the other's.
- `rm-model` removes named models only; there is no remove-everything command.
