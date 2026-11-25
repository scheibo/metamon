#!/usr/bin/env bash
# Build an ultra-compact, self-contained native darwin/arm64 .tar.zst bundle of
# Metamon TaurosV0 (gen1ou) for https://pkmn.ai.
set -euo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.build-cache"

MODEL_PATH=""
ONNX_THREADS="4"
SAMPLE="1"
TEMPERATURE="1.0"
WIN_MODE="logit"
WIN_OFFSET="0.0"
WIN_SCALE="2000.0"
WIN_SLOPE="1.0697"
WIN_INTERCEPT="-0.5143"
LOG_LEVEL="WARNING"
OUT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --onnx-threads) ONNX_THREADS="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --win-mode) WIN_MODE="$2"; shift 2 ;;
    --win-offset) WIN_OFFSET="$2"; shift 2 ;;
    --win-scale) WIN_SCALE="$2"; shift 2 ;;
    --win-slope) WIN_SLOPE="$2"; shift 2 ;;
    --win-intercept) WIN_INTERCEPT="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --out) OUT_PATH="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="${CACHE_DIR}/onnx/tauros_v0_fp16.onnx"
fi
if [[ -z "$OUT_PATH" ]]; then
  OUT_PATH="${REPO_ROOT}/dist/metamon-tauros-v0-gen1ou.tar.zst"
fi
mkdir -p "$(dirname "$OUT_PATH")" "$CACHE_DIR"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ONNX model not found at ${MODEL_PATH}. Run scripts/export_tauros_onnx.py first." >&2
  exit 2
fi

export PATH="/opt/homebrew/bin:${PATH}"

PY_VER="3.13"
PY_TARBALL="cpython-3.13.18+20261110-aarch64-apple-darwin-install_only_stripped.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.13.18%2B20261110-aarch64-apple-darwin-install_only_stripped.tar.gz"
PY_TAR_PATH="${CACHE_DIR}/${PY_TARBALL}"
PY_FULL_DIR="${CACHE_DIR}/py-darwin-full"

# Reuse cached python-build-standalone tarball from sibling foul-play repo if present
if [[ ! -f "$PY_TAR_PATH" && -f "${REPO_ROOT}/../foul-play/.build-cache/${PY_TARBALL}" ]]; then
  cp "${REPO_ROOT}/../foul-play/.build-cache/${PY_TARBALL}" "$PY_TAR_PATH"
fi

if [[ ! -f "${PY_FULL_DIR}/python/bin/python${PY_VER}" ]]; then
  if [[ ! -f "$PY_TAR_PATH" ]]; then
    echo "==> Downloading python-build-standalone (${PY_TARBALL})..."
    curl -fSL "$PY_URL" -o "${PY_TAR_PATH}.tmp"
    mv "${PY_TAR_PATH}.tmp" "$PY_TAR_PATH"
  fi
  rm -rf "$PY_FULL_DIR"
  mkdir -p "$PY_FULL_DIR"
  tar -xzf "$PY_TAR_PATH" -C "$PY_FULL_DIR"
fi

BUILD_PY="${PY_FULL_DIR}/python/bin/python${PY_VER}"

# 1. Stage minimal runtime packages (onnxruntime, numpy, orjson, termcolor, poke-env) for cp313 arm64
DEPS_STAGE="${CACHE_DIR}/runtime-deps-cp313"
if [[ ! -d "${DEPS_STAGE}/onnxruntime" || ! -d "${DEPS_STAGE}/numpy" || ! -d "${DEPS_STAGE}/orjson" || ! -d "${DEPS_STAGE}/termcolor" || ! -d "${DEPS_STAGE}/poke_env" ]]; then
  echo "==> Installing runtime dependencies into ${DEPS_STAGE}..."
  rm -rf "$DEPS_STAGE"
  mkdir -p "$DEPS_STAGE"
  if [[ -d "${CACHE_DIR}/export-venv/lib/python${PY_VER}/site-packages/onnxruntime" ]]; then
    # Copy already-installed exact cp313 arm64 wheels from export-venv
    SP="${CACHE_DIR}/export-venv/lib/python${PY_VER}/site-packages"
    cp -R "${SP}/onnxruntime" "${DEPS_STAGE}/onnxruntime"
    cp -R "${SP}/numpy" "${DEPS_STAGE}/numpy"
    cp -R "${SP}/orjson" "${DEPS_STAGE}/orjson"
    cp -R "${SP}/termcolor" "${DEPS_STAGE}/termcolor"
    cp -R "${SP}/poke_env" "${DEPS_STAGE}/poke_env"
  else
    "$BUILD_PY" -m pip install --no-deps --target "$DEPS_STAGE" \
      onnxruntime numpy orjson termcolor "poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git"
  fi
fi

# 2. Assemble stripped bundle in clean temporary staging directory
STAGE="$(mktemp -d "${CACHE_DIR}/stage-bundle.XXXXXX")"
trap 'chmod -R u+w "$STAGE" 2>/dev/null || true; rm -rf "$STAGE"' EXIT

echo "==> Staging runtime Python ${PY_VER}..."
mkdir -p "${STAGE}/python/bin" "${STAGE}/python/lib/python${PY_VER}/lib-dynload" "${STAGE}/site" "${STAGE}/model"
cp "${PY_FULL_DIR}/python/bin/python${PY_VER}" "${STAGE}/python/bin/python${PY_VER}"
# python-build-standalone links python3.13 statically; libpython*.dylib is unused (~15 MiB saved).

# Copy only needed C extension modules into lib-dynload (keep everything onnxruntime/numpy/stdlib need)
for ext in "${PY_FULL_DIR}/python/lib/python${PY_VER}/lib-dynload"/*.so; do
  base="$(basename "$ext")"
  case "$base" in
    _tkinter*|*test*|*curses*|*readline*|*dbm*|*gdbm*|*sqlite3*|*nis*|*ossaudiodev*|*audioop*)
      continue ;;
    *)
      cp "$ext" "${STAGE}/python/lib/python${PY_VER}/lib-dynload/" ;;
  esac
done

# Zip stdlib into python313.zip (dramatically cuts file count and unpack time)
echo "==> Zipping pruned Python ${PY_VER} stdlib..."
STDLIB_SRC="${PY_FULL_DIR}/python/lib/python${PY_VER}"
"$BUILD_PY" - "$STDLIB_SRC" "${STAGE}/python/lib/python313.zip" <<'PY'
import os, sys, zipfile

src_root, zip_out = sys.argv[1], sys.argv[2]
SKIP_TOP = {
    "test", "tests", "tkinter", "turtledemo", "turtle.py", "idlelib",
    "pydoc_data", "ensurepip", "venv", "sqlite3", "unittest", "doctest.py",
    "lib2to3", "distutils", "curses", "dbm", "xmlrpc", "http", "imaplib.py",
    "poplib.py", "smtplib.py", "ftplib.py", "telnetlib.py", "nntplib.py",
    "cgi.py", "cgitb.py", "wsgiref", "antigravity.py", "this.py", "lib-dynload",
    "site-packages", "config-3.13-darwin", "EXTERNALLY-MANAGED", "__pycache__",
}

count = 0
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for root, dirs, files in os.walk(src_root):
        rel_root = os.path.relpath(root, src_root)
        if rel_root == ".":
            dirs[:] = [d for d in sorted(dirs) if d not in SKIP_TOP]
        else:
            dirs[:] = [d for d in sorted(dirs) if d != "__pycache__" and d != "tests" and d != "test"]
        for fname in sorted(files):
            if rel_root == "." and fname in SKIP_TOP:
                continue
            if not fname.endswith(".py"):
                continue
            full = os.path.join(root, fname)
            arc = fname if rel_root == "." else os.path.join(rel_root, fname)
            zf.write(full, arc)
            count += 1
print(f"    packed {count} stdlib .py files into python313.zip")
PY

# 3. Copy & prune onnxruntime, numpy, orjson
echo "==> Copying and pruning onnxruntime, numpy, orjson..."
cp -R "${DEPS_STAGE}/onnxruntime" "${STAGE}/site/onnxruntime"
# Remove duplicate 34 MiB dylib (onnxruntime_pybind11_state.so is self-contained) + unused subpkgs
rm -f "${STAGE}/site/onnxruntime/capi"/libonnxruntime*.dylib
rm -rf "${STAGE}/site/onnxruntime"/{transformers,quantization,tools,backend,datasets}
rm -f "${STAGE}/site/onnxruntime/ThirdPartyNotices.txt" "${STAGE}/site/onnxruntime/LICENSE"

cp -R "${DEPS_STAGE}/numpy" "${STAGE}/site/numpy"
rm -rf "${STAGE}/site/numpy"/{tests,f2py,testing,typing,polynomial,ma,doc,_pyinstaller,distutils}
find "${STAGE}/site/numpy" -type d \( -name tests -o -name include -o -name __pycache__ \) -exec rm -rf {} + 2>/dev/null || true
find "${STAGE}/site/numpy" -type f \( -name '*.pyi' -o -name '*.pxd' -o -name '*.h' -o -name '*.a' -o -name '*.c' \) -delete

cp -R "${DEPS_STAGE}/orjson" "${STAGE}/site/orjson"
find "${STAGE}/site/orjson" -type f -name '*.pyi' -delete

cp -R "${DEPS_STAGE}/termcolor" "${STAGE}/site/termcolor"
find "${STAGE}/site/termcolor" -type f -name '*.pyi' -delete

# 4. Copy minimal Gen 1-only poke_env (zero POKE_LOOP / websockets / requests / gymnasium)
echo "==> Staging minimal Gen 1 poke_env..."
mkdir -p "${STAGE}/site/poke_env/data/static"/{moves,pokedex,typechart} \
         "${STAGE}/site/poke_env/player" \
         "${STAGE}/site/poke_env-0.8.3.3.dist-info"
printf 'Metadata-Version: 2.1\nName: poke-env\nVersion: 0.8.3.3\n' \
  > "${STAGE}/site/poke_env-0.8.3.3.dist-info/METADATA"
cp "${DEPS_STAGE}/poke_env/exceptions.py" "${STAGE}/site/poke_env/exceptions.py"
cp "${DEPS_STAGE}/poke_env/stats.py" "${STAGE}/site/poke_env/stats.py"
cp -R "${DEPS_STAGE}/poke_env/environment" "${STAGE}/site/poke_env/environment"
cp -R "${DEPS_STAGE}/poke_env/teambuilder" "${STAGE}/site/poke_env/teambuilder"
cp "${DEPS_STAGE}/poke_env/player/battle_order.py" "${STAGE}/site/poke_env/player/battle_order.py"
cp "${DEPS_STAGE}/poke_env/data/__init__.py" "${STAGE}/site/poke_env/data/__init__.py"
cp "${DEPS_STAGE}/poke_env/data/gen_data.py" "${STAGE}/site/poke_env/data/gen_data.py"
cp "${DEPS_STAGE}/poke_env/data/normalize.py" "${STAGE}/site/poke_env/data/normalize.py"
cp "${DEPS_STAGE}/poke_env/data/replay_template.py" "${STAGE}/site/poke_env/data/replay_template.py"
cp "${DEPS_STAGE}/poke_env/data/static/replay_template.html" "${STAGE}/site/poke_env/data/static/replay_template.html"
cp "${DEPS_STAGE}/poke_env/data/static/natures.json" "${STAGE}/site/poke_env/data/static/natures.json"
printf '{}' > "${STAGE}/site/poke_env/data/static/learnset.json"
cp "${DEPS_STAGE}/poke_env/data/static/moves/gen1moves.json" "${STAGE}/site/poke_env/data/static/moves/gen1moves.json"
cp "${DEPS_STAGE}/poke_env/data/static/pokedex/gen1pokedex.json" "${STAGE}/site/poke_env/data/static/pokedex/gen1pokedex.json"
cp "${DEPS_STAGE}/poke_env/data/static/typechart/gen1typechart.json" "${STAGE}/site/poke_env/data/static/typechart/gen1typechart.json"

cat > "${STAGE}/site/poke_env/__init__.py" <<'PY'
"""Minimal poke_env surface for offline Metamon Gen 1 OU inference."""
import poke_env.environment as environment
import poke_env.exceptions as exceptions
import poke_env.stats as stats
from poke_env.data import gen_data, to_id_str
from poke_env.exceptions import ShowdownException
from poke_env.stats import compute_raw_stats

__all__ = [
    "ShowdownException",
    "compute_raw_stats",
    "environment",
    "exceptions",
    "gen_data",
    "stats",
    "to_id_str",
]
PY

cat > "${STAGE}/site/poke_env/player/__init__.py" <<'PY'
"""Minimal poke_env.player surface (BattleOrder + Player.create_order, no POKE_LOOP)."""
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder, ForfeitBattleOrder


class Player:
    @staticmethod
    def create_order(
        order,
        mega: bool = False,
        z_move: bool = False,
        dynamax: bool = False,
        terastallize: bool = False,
        move_target: int = 0,
    ) -> BattleOrder:
        return BattleOrder(
            order,
            mega=mega,
            z_move=z_move,
            dynamax=dynamax,
            terastallize=terastallize,
            move_target=move_target,
        )


__all__ = ["BattleOrder", "DefaultBattleOrder", "ForfeitBattleOrder", "Player"]
PY

# 5. Copy minimal Gen 1-only metamon runtime + ONNX model
echo "==> Staging minimal Metamon Gen 1 runtime + ONNX model..."
mkdir -p "${STAGE}/metamon/env" \
         "${STAGE}/metamon/tokenizer" \
         "${STAGE}/metamon/backend/replay_parser" \
         "${STAGE}/metamon/backend/showdown_dex/static"/{moves,pokemon,typechart}

cp "${REPO_ROOT}/metamon/__init__.py" "${STAGE}/metamon/__init__.py"
cp "${REPO_ROOT}/metamon/config.py" "${STAGE}/metamon/config.py"
cp "${REPO_ROOT}/metamon/interface.py" "${STAGE}/metamon/interface.py"
cp "${REPO_ROOT}/metamon/pkmn_ai.py" "${STAGE}/metamon/pkmn_ai.py"
cp "${REPO_ROOT}/metamon/env/__init__.py" "${STAGE}/metamon/env/__init__.py"
cp "${REPO_ROOT}/metamon/env/metamon_battle.py" "${STAGE}/metamon/env/metamon_battle.py"
cp "${REPO_ROOT}/metamon/tokenizer/__init__.py" "${STAGE}/metamon/tokenizer/__init__.py"
cp "${REPO_ROOT}/metamon/tokenizer/tokenizer.py" "${STAGE}/metamon/tokenizer/tokenizer.py"
cp "${REPO_ROOT}/metamon/tokenizer/DefaultObservationSpace-v1.json" \
   "${STAGE}/metamon/tokenizer/DefaultObservationSpace-v1.json"
cp "${REPO_ROOT}/metamon/backend/__init__.py" "${STAGE}/metamon/backend/__init__.py"
cp "${REPO_ROOT}/metamon/backend/replay_parser"/*.py "${STAGE}/metamon/backend/replay_parser/"
rm -f "${STAGE}/metamon/backend/replay_parser/backward.py" "${STAGE}/metamon/backend/replay_parser/run.py"
cp "${REPO_ROOT}/metamon/backend/showdown_dex/__init__.py" "${STAGE}/metamon/backend/showdown_dex/__init__.py"
cp "${REPO_ROOT}/metamon/backend/showdown_dex/dex.py" "${STAGE}/metamon/backend/showdown_dex/dex.py"
cp "${REPO_ROOT}/metamon/backend/showdown_dex/static/natures.json" \
   "${STAGE}/metamon/backend/showdown_dex/static/natures.json"
cp "${REPO_ROOT}/metamon/backend/showdown_dex/static/moves/gen1moves.json" \
   "${STAGE}/metamon/backend/showdown_dex/static/moves/gen1moves.json"
cp "${REPO_ROOT}/metamon/backend/showdown_dex/static/pokemon/gen1pokedex.json" \
   "${STAGE}/metamon/backend/showdown_dex/static/pokemon/gen1pokedex.json"
cp "${REPO_ROOT}/metamon/backend/showdown_dex/static/typechart/gen1typechart.json" \
   "${STAGE}/metamon/backend/showdown_dex/static/typechart/gen1typechart.json"

# Minify static JSON files to shrink unpacked footprint
"$BUILD_PY" - "${STAGE}/site/poke_env/data/static" "${STAGE}/metamon/backend/showdown_dex/static" "${STAGE}/metamon/tokenizer" <<'PY'
import json, pathlib, sys
for root in sys.argv[1:]:
    for p in pathlib.Path(root).rglob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
PY

MODEL_BASENAME="$(basename "$MODEL_PATH")"
cp "$MODEL_PATH" "${STAGE}/model/${MODEL_BASENAME}"

# Remove any stray __pycache__ / .DS_Store files
find "$STAGE" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

# 6. Strip and ad-hoc codesign every Mach-O binary (mandatory on Apple Silicon!)
echo "==> Stripping and ad-hoc codesigning Mach-O binaries..."
while IFS= read -r -d '' bin; do
  chmod u+w "$bin"
  strip -x "$bin" 2>/dev/null || true
  codesign --force --sign - "$bin" >/dev/null 2>&1
  codesign --verify --strict "$bin" >/dev/null 2>&1
done < <(find "$STAGE" -type f \( -name '*.so' -o -name '*.dylib' -o -path '*/python/bin/python*' \) -print0)

# 7. Write entrypoint script `./run`
cat > "${STAGE}/run" <<EOF
#!/bin/bash
set -euo pipefail
DIR="\$(CDPATH= cd -- "\$(dirname -- "\$0")" && pwd)"
export PYTHONHOME="\${DIR}/python"
export PYTHONPATH="\${DIR}/python/lib/python313.zip:\${DIR}/python/lib/python${PY_VER}/lib-dynload:\${DIR}/site:\${DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export METAMON_MODEL_PATH="\${METAMON_MODEL_PATH:-\${DIR}/model/${MODEL_BASENAME}}"
export METAMON_ONNX_THREADS="\${METAMON_ONNX_THREADS:-${ONNX_THREADS}}"
export METAMON_SAMPLE="\${METAMON_SAMPLE:-${SAMPLE}}"
export METAMON_TEMPERATURE="\${METAMON_TEMPERATURE:-${TEMPERATURE}}"
export METAMON_WIN_MODE="\${METAMON_WIN_MODE:-${WIN_MODE}}"
export METAMON_WIN_OFFSET="\${METAMON_WIN_OFFSET:-${WIN_OFFSET}}"
export METAMON_WIN_SCALE="\${METAMON_WIN_SCALE:-${WIN_SCALE}}"
export METAMON_WIN_SLOPE="\${METAMON_WIN_SLOPE:-${WIN_SLOPE}}"
export METAMON_WIN_INTERCEPT="\${METAMON_WIN_INTERCEPT:-${WIN_INTERCEPT}}"
export METAMON_LOG_LEVEL="\${METAMON_LOG_LEVEL:-${LOG_LEVEL}}"
exec "\${DIR}/python/bin/python${PY_VER}" -P -W ignore::SyntaxWarning -m metamon.pkmn_ai "\$@"
EOF
chmod 755 "${STAGE}/run"

# 8. Smoke-test from a read-only copy before archiving
echo "==> Smoke-testing bundle from read-only rootfs..."
RO_CHECK="${CACHE_DIR}/ro-smoke-check"
chmod -R u+w "$RO_CHECK" 2>/dev/null || true
rm -rf "$RO_CHECK"
cp -R "$STAGE" "$RO_CHECK"
chmod -R a-w "$RO_CHECK"

"$BUILD_PY" - "${RO_CHECK}/run" <<'PY'
import json, subprocess, sys

run_bin = sys.argv[1]
proc = subprocess.Popen(
    [run_bin],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)
assert proc.stdin and proc.stdout and proc.stderr
ready = proc.stdout.readline().strip()
assert ready == "/ready", f"Expected /ready, got {ready!r}; stderr={proc.stderr.read()!r}"

req = {
    "active": [{
        "moves": [
            {"move": "Body Slam", "id": "bodyslam", "pp": 24, "maxpp": 24, "target": "normal", "disabled": False},
            {"move": "Hyper Beam", "id": "hyperbeam", "pp": 8, "maxpp": 8, "target": "normal", "disabled": False},
            {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
            {"move": "Blizzard", "id": "blizzard", "pp": 8, "maxpp": 8, "target": "allAdjacentFoes", "disabled": False},
        ]
    }],
    "side": {
        "name": "P1",
        "id": "p1",
        "pokemon": [
            {"ident": "p1: Tauros", "details": "Tauros", "condition": "353/353", "active": True, "stats": {"atk": 298, "def": 288, "spa": 238, "spd": 238, "spe": 318}, "moves": ["bodyslam", "hyperbeam", "earthquake", "blizzard"], "baseAbility": "noability", "item": ""},
            {"ident": "p1: Chansey", "details": "Chansey", "condition": "703/703", "active": False, "stats": {"atk": 108, "def": 108, "spa": 308, "spd": 308, "spe": 198}, "moves": ["icebeam", "softboiled", "thunderwave", "seismictoss"], "baseAbility": "noability", "item": ""},
        ]
    },
    "rqid": 1,
}
lines = [
    "|utm|Tauros|||noability|bodyslam,hyperbeam,earthquake,blizzard|||||||",
    f"|request|{json.dumps(req)}",
    "|player|p1|P1||",
    "|player|p2|P2||",
    "|teamsize|p1|6",
    "|teamsize|p2|6",
    "|gametype|singles",
    "|gen|1",
    "|tier|[Gen 1] OU",
    "|start",
    "|switch|p1a: Tauros|Tauros|353/353",
    "|switch|p2a: Alakazam|Alakazam|100/100",
    "|turn|1",
]
for l in lines:
    proc.stdin.write(l + "\n")
proc.stdin.flush()

info_line = proc.stdout.readline().strip()
choice_line = proc.stdout.readline().strip()
proc.stdin.close()
stderr = proc.stderr.read()
proc.wait(timeout=5)

assert info_line.startswith("/info "), f"Expected /info, got {info_line!r}; stderr={stderr!r}"
info = json.loads(info_line[6:])
assert 0.0 <= info["win"] <= 1.0, f"Invalid win: {info}"
assert isinstance(info.get("policy"), dict) and 1 <= len(info["policy"]) <= 8, f"Invalid policy: {info}"
assert choice_line.startswith(("move ", "switch ")), f"Invalid choice: {choice_line!r}"
assert stderr.strip() == "", f"Unexpected stderr: {stderr!r}"
print(f"    RO smoke PASS: {info_line} -> {choice_line}")
PY
chmod -R u+w "$RO_CHECK"
rm -rf "$RO_CHECK"

# 9. Archive with macOS xattr suppression + zstd -19
echo "==> Archiving -> ${OUT_PATH}..."
UNPACKED_BYTES="$(du -sk "$STAGE" | awk '{print $1 * 1024}')"
ENTRY_COUNT="$(find "$STAGE" | wc -l | tr -d ' ')"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -C "$STAGE" -cf - . \
  | zstd -19 -T0 -f -o "$OUT_PATH"

COMP_BYTES="$(stat -f%z "$OUT_PATH")"
SHA256="$(shasum -a 256 "$OUT_PATH" | awk '{print $1}')"
echo "==> Built ${OUT_PATH}"
echo "    Compressed: $((COMP_BYTES / 1024 / 1024)) MiB (${COMP_BYTES} bytes)"
echo "    Unpacked:   $((UNPACKED_BYTES / 1024 / 1024)) MiB (${UNPACKED_BYTES} bytes)"
echo "    Entries:    ${ENTRY_COUNT}"
echo "    SHA-256:    ${SHA256}"
