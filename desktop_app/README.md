# OserAI Trainer (desktop app)

A friendly front-end for `agent/train_agent.py`: fill in a coordinator URL
and a training run ID, hit **Start contributing**, and this machine
registers as a DiLoCo training node and starts pulling rounds -- the same
"install once, contribute compute in the background" shape as
BOINC/Folding@home. Closing the window minimizes to the system tray rather
than quitting, so a round in progress isn't interrupted.

All the actual training logic (registration, heartbeat, round polling,
local LoRA/scratch-GPT steps, delta upload) lives in `agent/train_agent.py`
and `training/*` -- this app is a GUI wrapper around that tested code, not
a separate implementation.

## Running from source

```bash
pip install -r requirements-desktop.txt
python -m desktop_app.main
```

## Packaging a standalone app

```bash
pip install -r requirements-desktop.txt
cd desktop_app
pyinstaller build.spec
```

Produces `desktop_app/dist/OserAI-Trainer/` -- a folder containing a native
launcher (`OserAI-Trainer.exe` on Windows, `OserAI-Trainer` on
macOS/Linux) plus a bundled Python runtime and all dependencies. Contributors
run the launcher directly; no separate Python install needed on their end.

**Build once per OS you want to ship for** -- PyInstaller doesn't
cross-compile. A build run on macOS produces a macOS app; Windows and Linux
builds need to be run on those platforms (or in CI runners for them). Only
the macOS build has actually been produced and run in this repo so far --
the Windows/Linux steps above are the same command, just not yet verified
here for lack of those machines.

## What a contributor needs to fill in

- **Coordinator URL** -- e.g. `http://coordinator.example.com:8000`.
- **Node name** -- defaults to the machine's hostname.
- **Run ID** -- which training run to join; get this from whoever is
  running the coordinator (`POST /training/runs` returns it, or check the
  dashboard's Training Runs section).
- **Shard index / Num shards** -- which slice of the training data this
  machine trains on. If you're one of 4 known contributor machines, that's
  shard index 0-3 out of 4 shards; the coordinator operator tells you this.
- **Device** -- auto-populated with what's actually available
  (`cuda`/`mps`/`cpu`); pick the fastest one present.

Settings are saved to `~/.oserai_trainer/settings.json` and restored next
launch.
