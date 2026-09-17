#!/usr/bin/env bash
# Syncs this repo to a Hetzner server and (re)starts the coordinator there.
# The GitHub repo is private, so cloud-init can't just `git clone` it on
# first boot -- this script is how code actually gets onto the box, both
# for the first deploy and every redeploy after.
#
# Usage: deploy/deploy.sh <server-ip-or-hostname> [ssh-user]
#
# Prereqs: the server has already been created (Hetzner console, Ubuntu
# 24.04, deploy/cloud-init.yaml pasted into "Cloud config"), and has
# finished its first boot (Docker installed, firewall up).

set -euo pipefail

HOST="${1:?usage: deploy.sh <server-ip-or-hostname> [ssh-user]}"
SSH_USER="${2:-root}"
REMOTE_DIR="/opt/oserai"

cd "$(dirname "$0")/.."

echo "==> syncing repo to ${SSH_USER}@${HOST}:${REMOTE_DIR}"
rsync -az --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm' \
    --exclude='out/' \
    --exclude='*.pt' \
    --exclude='training/data/tinystories.txt' \
    --exclude='training/data/dolly15k_chat.txt' \
    --exclude='desktop_app/build/' --exclude='desktop_app/dist/' \
    --exclude='node_config.yaml' --exclude='train_node_config.yaml' \
    ./ "${SSH_USER}@${HOST}:${REMOTE_DIR}/"

echo "==> building + (re)starting the coordinator"
ssh "${SSH_USER}@${HOST}" "cd ${REMOTE_DIR}/deploy && docker compose -f docker-compose.prod.yaml up -d --build"

echo "==> done. Tail logs with:"
echo "    ssh ${SSH_USER}@${HOST} 'cd ${REMOTE_DIR}/deploy && docker compose -f docker-compose.prod.yaml logs -f'"
