# JarvisLabs runbook — launch to teardown

Target instance: **A100 80GB · 28 vCPU · 112 GB RAM · 100 GB storage · dynamic IP**
(~$1.49/hr ≈ ₹142/hr). Full suite of 13 arms ≈ 15–20 h ≈ **₹2,500–3,200**.

---

## 1. Add your SSH key — before launching

The key must be registered on the account *before* the instance boots; a running VM
picks up keys at launch.

**a. Check for an existing key, or make one:**

```bash
ls ~/.ssh/id_ed25519.pub          # already have one? skip the next line
ssh-keygen -t ed25519 -C "gaurigst1970@gmail.com"
```

Accept the default path, set a passphrase if you want one.

**b. Copy the PUBLIC key** (the `.pub` file — never the private one):

```bash
pbcopy < ~/.ssh/id_ed25519.pub    # macOS; puts it on the clipboard
```

**c. Register it** at **<https://jarvislabs.ai/settings/ssh>** → paste → save.
The whole one-line `ssh-ed25519 AAAA... you@host` string goes in.

> Only ever paste the `.pub` contents. If you accidentally paste `~/.ssh/id_ed25519`
> (no extension), that is your private key — rotate it immediately.

---

## 2. Launch — a Template, not a VM

JarvisLabs offers two products on the same GPUs at the same price:

| | **Template** ← use this | VM |
|---|---|---|
| what it is | pre-built container; they manage OS, drivers, CUDA, runtime | clean VM, full root, your own kernel/Docker |
| PyTorch + CUDA | **preinstalled** | you install it yourself |
| boot | ~2 s | ~90 s |

**Launch the PyTorch Template.** This is load-bearing, not cosmetic: `requirements.txt`
deliberately leaves torch unpinned and `jarvislabs_setup.sh` builds its venv with
`--system-site-packages`, both so we inherit the image's CUDA-matched torch. A bare VM
has no torch to inherit, so pip would pull a multi-GB wheel that may not match the
driver. We need no root, Docker or systemd, so the VM buys nothing here.

Dashboard → **Create instance**:

| setting | value | why |
|---|---|---|
| Template | **PyTorch** | avoid task-specific templates; we install the rest |
| GPU | **A100 80GB** | bandwidth helps the generation-heavy eval; 40GB also fits this 1B model |
| GPU count | **1** | nothing here is multi-GPU; use more only to run arms in parallel |
| Storage | **100 GB** | actual use ≈ 25–30 GB, so this is comfortable |
| Startup script | *(none)* | we bootstrap explicitly in step 4 |

---

## 3. Connect

Copy the SSH command from the dashboard (copy icon next to **SSH**). Templates connect
through a shared gateway on a **per-instance port**, so it looks like:

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@sshd.jarvislabs.ai
```

The user is **`root`** and the port is what identifies your instance — always copy the
current command from the dashboard rather than reusing an old one, since **the port can
change when an instance is resumed**. For a key in a non-default location, add
`-i ~/.ssh/my_key`.

No password — auth uses the key from step 1. If the key is refused, the usual cause is
that it was registered *after* the instance booted: re-add it and restart the instance.

---

## 4. Bootstrap

```bash
curl -sSL https://raw.githubusercontent.com/gadmin7/mwp_ai4math_icml_v2/main/cloud/jarvislabs_setup.sh -o setup.sh
bash setup.sh
```

It clones the repo, builds a venv **under `/home`** (see §7), installs requirements,
prompts for `hf auth login`, and then runs both verification passes — the smoke test
and a full end-to-end dry run — *before* you spend money on a real job.

Llama-3.2-1B-Instruct is **gated**. The `GT1999` account already has access, so
`hf auth login` on the box with a token that carries read access to gated repos plus
write access (checkpoints push to `GT1999/*`) is all that's needed. If you generate a
*new* fine-grained token for the instance, make sure both boxes are ticked — a
read-only token fails at the first push, i.e. only after a stage has finished training.

---

## 5. Run the baselines

Every shell needs the environment first:

```bash
source /home/mwp-venv/bin/activate
export HF_HOME=/home/.cache/huggingface
export HF_TOKEN=$(hf auth token)
cd /home/mwp_ai4math_icml_v2 && mkdir -p runs
```

> Paths are under `/home`, not `~`, deliberately — see §7.

**Run everything, in order, in one tmux session** (recommended — this is a 15–20 h job
you shouldn't have to babysit):

```bash
tmux new -s mwp 'bash scripts/run_all.sh 2>&1 | tee runs/run_all.log'
# detach: Ctrl-b then d      reattach: tmux attach -t mwp      list: tmux ls
```

`run_all.sh` runs the arms sequentially in the order below, logs each to `runs/bN.log`,
times each one, and records any failure in `runs/failures.txt` **without aborting the
rest** — losing twelve arms because arm three hit a transient Hub error would be worse
than finishing and retrying that one. Override the order with `ORDER="6 7 8"`.

Or drive a single arm yourself:

```bash
tmux new -s b9 'python3 scripts/run_baseline.py --config configs/baseline9.yaml 2>&1 | tee runs/b9.log'
```

**Recommended order** — cheap/decisive arms first, so a problem surfaces before the
expensive ones and each block answers something before the next begins:

| block | arms | why first |
|---|---|---|
| 1 | `9`, `10` | single-stage, fastest; settles the Table 2 scaling confound |
| 2 | `1`, `2` | confirms the pipeline on real data end-to-end |
| 3 | `6`, `7`, `8` | the capacity-matched shrink/expand/constant triad |
| 4 | `11`, `12`, `13` | curriculum-ordering controls |
| 5 | `3`, `4`, `5` | remaining paper reproductions |

Run them **sequentially on one GPU**. Two training jobs on the same card will contend
for VRAM and slow each other down; to parallelise, launch a multi-GPU instance and pin
one arm per GPU with `CUDA_VISIBLE_DEVICES`.

---

## 6. Monitoring

```bash
tail -f runs/b9.log
watch -n5 nvidia-smi          # GPU utilisation + memory
```

Sanity checks worth watching for in the log:

- `level train val test` table, then **`val fraction of train+val = 0.050`**
- `stage partition strategy: ...` on multi-stage arms
- `[best model] restored adapter from step N` at the end of each stage — if this says
  *"no eval improvement recorded"*, the stage never improved and kept its final weights
- `pushed GT1999/mwp-v2-llama1b-bN-stageM` per stage

If VRAM runs tight, drop `--batch-size` (e.g. `--batch-size 16`); nothing else needs
changing.

---

## 7. Pause / resume — read before you pause

Two behaviours to plan around:

1. **Only `/home` persists.** Anything installed globally (system pip, `apt`) is lost on
   resume. Note the trap: templates log you in as **root**, so `$HOME` is `/root`, which
   is *not* persistent — a venv at `~/mwp-venv` would be wiped on every resume. That is
   why the bootstrap hardcodes `/home/mwp-venv` and `/home/.cache/huggingface` instead of
   using `$HOME`. A resumed box therefore keeps both its dependencies and its multi-GB
   model downloads. After a resume just re-source the environment (§5); only re-run
   `setup.sh` if imports fail.
2. **Connection details can change on resume.** Re-copy the SSH command from the
   dashboard — the port is what identifies the instance. Training inside tmux is
   unaffected by your disconnection.

**Pausing still bills storage.** Deleting is irreversible and takes `/home` with it —
so confirm your checkpoints are on the Hub first (§8).

---

## 8. Before teardown

```bash
ls runs/*/                            # stage dirs + <arm>-log_history.json
hf auth whoami
```

Confirm every arm's stages are on the Hub — `GT1999/mwp-v2-llama1b-b{N}-stage{1..5}` —
since those adapters, not the local `runs/`, are what the evaluation and the
weight-geometry analysis load. Pull anything you want to keep locally:

```bash
# from your Mac
scp -P <port> -r root@sshd.jarvislabs.ai:/home/mwp_ai4math_icml_v2/runs ./runs-from-cloud
```

`runs/*-log_history.json` holds the per-stage loss curves and step counts — worth
keeping for the paper's compute-cost table. Then **delete** the instance (pausing
keeps charging for storage).

---

## Cost notes

- Billing is per-minute; deleting stops it, pausing does not stop storage charges.
- Spot instances run up to ~56% cheaper but can be paused when demand rises — fine for
  tmux'd jobs you can resume, risky if you want an unattended overnight run.
- JarvisLabs bills in USD, so an Indian card will typically add ~1–3.5% forex markup.
