#!/bin/bash
# Publish viewer/ to Cloudflare R2 so the renders are reachable from any network.
#
# WHY R2 AND NOT PORT-FORWARDING THE NAS: forwarding 80/443 puts the DSM login
# page on the public internet, and Synology boxes are actively scanned (DeadBolt
# ransomware, the 2021 StealthWorker campaign). Serving static files from object
# storage means there is no NAS in the request path at all -- nothing to attack,
# nothing to keep patched, and the 507 MB is inside R2's 10 GB free tier with
# zero egress fees.
#
# R2 also natively answers HTTP Range requests, which is what the <video>
# overlay needs to seek. That was the whole reason scripts/serve.py exists and
# why the NAS had to run nginx rather than python -m http.server.
#
# CREDENTIALS ARE READ FROM THE ENVIRONMENT AND NEVER STORED HERE.
#   export R2_ACCOUNT_ID=...
#   export R2_ACCESS_KEY_ID=...
#   export R2_SECRET_ACCESS_KEY=...
#   export R2_BUCKET=fpv-splat
#
# Then:  bash scripts/publish_r2.sh
set -euo pipefail
: "${R2_ACCOUNT_ID:?set R2_ACCOUNT_ID}"
: "${R2_ACCESS_KEY_ID:?set R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?set R2_SECRET_ACCESS_KEY}"
BUCKET="${R2_BUCKET:-fpv-splat}"
SRC=~/Desktop/fpv-splat/viewer

command -v rclone >/dev/null || { echo "rclone missing: brew install rclone"; exit 1; }

# Exactly the 12 files + video/ that the working NAS deployment serves. Uploading
# the whole viewer/ directory would push ~700 MB of *_clean.splat and *_path.splat
# that the web build does not reference.
FILES=(index.html main.js ghosts.js
       cameras.json site_traj.json site_web.splat
       aug_cameras.json aug_traj.json aug_web.splat
       aug2_cameras.json aug2_traj.json aug2_web.splat)

export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export RCLONE_CONFIG_R2_ACL=private        # public access comes from the bucket, not per-object

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
for f in "${FILES[@]}"; do cp "$SRC/$f" "$STAGE/"; done
cp -R "$SRC/video" "$STAGE/video"

echo "=== staged $(du -sh "$STAGE" | cut -f1) ==="

# Big immutable assets: cache hard. Splats and proxies never change in place --
# a new scene gets a new filename.
rclone copy "$STAGE" "R2:$BUCKET" \
  --include "*.splat" --include "video/**" \
  --header-upload "Cache-Control: public, max-age=31536000, immutable" \
  --transfers 4 --checkers 8 --s3-chunk-size 32M --progress

# Code and metadata: revalidate, so an edit actually ships.
rclone copy "$STAGE" "R2:$BUCKET" \
  --include "*.html" --include "*.js" --include "*.json" \
  --header-upload "Cache-Control: public, max-age=60" \
  --transfers 4 --progress

echo
echo "=== uploaded. Verify: ==="
echo "  rclone ls R2:$BUCKET | wc -l     # expect 34"
B="https://pub-<hash>.r2.dev"
echo "R2 serves no directory index, so every link names index.html explicitly."
echo "main.js resolves ?url= against the page location and derives the trajectory"
echo "from the splat prefix (aug2_web.splat -> aug2_traj.json), but ?cameras="
echo "must be given for anything other than the May scene, whose cameras file is"
echo "the bare cameras.json default."
echo
echo "  May site  : $B/index.html?url=site_web.splat"
echo "  Hilltop   : $B/index.html?url=aug_web.splat&cameras=aug_cameras.json"
echo "  Cemetery  : $B/index.html?url=aug2_web.splat&cameras=aug2_cameras.json"
