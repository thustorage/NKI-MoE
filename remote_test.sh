#!/bin/bash

################################################################################
# Remote Test Script for NKI-MOE Project - AI Agent Helper
################################################################################
#
# Purpose:
#   This script is designed for AI agents to perform remote testing on a
#   remote server while running on a local machine.
#
# Architecture:
#   - Local Server: Where the code agent operates and modifies code
#   - Remote Server (neuron): Where actual tests are executed
#   - Git Repository: Synchronization mechanism between servers
#
# Workflow:
#   1. Stage and commit local changes on specified branch
#   2. Push changes to remote repository
#   3. SSH into remote server (configured via REMOTE_HOST)
#   4. Pull latest changes in REMOTE_PROJECT_DIR directory
#   5. Execute specified test command
#   6. Return output to local agent for analysis and modification
#
# Prerequisites:
#   - Git repository initialized on specified branch
#   - SSH config with remote server configured (configured via REMOTE_HOST)
#   - Remote server: project directory with git repository cloned (configured via REMOTE_PROJECT_DIR)
#   - Remote server: git pull must work without authentication prompts
#
# Usage:
#   ./remote_test.sh "<command_to_run_on_remote>"
#   ./remote_test.sh --push --commit-message "your message" "<command_to_run_on_remote>"
#   ./remote_test.sh --push --qwen-script qwen_with_nki_original.py "<command_to_run_on_remote>"
#   ./remote_test.sh --output results/foo.txt "<command_to_run_on_remote>"
#   ./remote_test.sh --nkilib-mode bundled "<command_to_run_on_remote>"
#
# Examples:
#   ./remote_test.sh "python test.py"
#   ./remote_test.sh --push "python test.py"
#   ./remote_test.sh "pytest -v tests/"
#   ./remote_test.sh "python -c 'print(\"Hello from remote!\")'"
#
# Common main.py commands:
#   MODEL_ARGS is auto-injected as env var on remote (see configuration variables at top of file).
#
#   --mode: generate | validate | validate-baseline | validate-nki | evaluate_single
#   --enable-nki: enable NKI kernels (loads qwen_with_nki module)
#   --num-hidden-layers N: limit to N layers (default: all)
#   --fused-qkv: required for QKV NKI kernel
#   --save-sharded-checkpoint: save sharded checkpoints after compilation for potential reuse with --skip-sharding
#   Custom NKI kernel flags: --custom-qkv-nki-kernel-enabled  --custom-o-proj-nki-kernel-enabled  --custom-moe-fused-nki-kernel-enabled
#   NXD framework fused flags: --use-nxd-fuse-moe  --use-nxd-fuse-attn
#   --skip-sharding: skip pre-sharding checkpoints during compile. Uses existing sharded weights
#         from a previous compile run. Only works when the weight key structure matches (i.e. same
#         fused-qkv setting). The fuzzy weight loader in NeuronQwen3MoeForCausalLM.load_weights()
#         auto-remaps wrapper-level key differences (self_attn.attn.* <-> self_attn.*, mlp.moe.* <-> mlp.*)
#         so --skip-sharding can be used across different NKI on/off configs AS LONG AS fused-qkv is the same.
#         Cannot handle tensor fusion changes (fused_qkv <-> separate q/k/v_proj).
#   --token-generation-buckets N: set TKG bucket size (default 1024, must be <= seq-len, e.g. 640)
#
#   NOTE: --mode validate passes NKI flags to baseline too. MoE baseline has dtype bug,
#         use validate-baseline + validate-nki separately for MoE.
#
# Options:
#   --push              Stage, commit, and push local changes to remote
#                       repository before running the command. Without this flag,
#                       the script assumes the remote server already has the latest
#                       code and skips git synchronization entirely.
#   --commit-message    Commit message used when --push needs to create a commit.
#   --qwen-script       Model module/file to pass through to main.py as --qwen.
#                       Accepts either module name or *.py filename.
#   --output            Save local combined output log to the given path.
#   --nkilib-mode       Choose nkilib import behavior on remote: auto | bundled.
#                       auto: do not force bundled nkilib.
#                       bundled: export NKILIB_FORCE_BUNDLED_LIBRARY=1.
#                       default is bundled on SDK 2.29 because nkilib_src still
#                       imports legacy APIs such as nki.tensor.
#
# Notes:
#   - This script creates a normal commit when local changes exist
#   - Remote sync uses git pull --ff-only
#   - Output from remote command is directly displayed
#
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_REMOTE_PROCESSES_SCRIPT="$SCRIPT_DIR/check_remote_repo_processes.sh"

################################################################################
# Configuration Variables - Loaded from .remote_test.config
################################################################################
CONFIG_FILE="$SCRIPT_DIR/.remote_test.config"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Create one with the following variables:"
    echo "  REMOTE_HOST, REMOTE_PROJECT_DIR, REMOTE_VENV, REMOTE_MODEL_PATH,"
    echo "  REMOTE_COMPILED_MODEL_PATH, NEURON_PLATFORM, BRANCH, BUNDLE_REMOTE_PATH"
    exit 1
fi
source "$CONFIG_FILE"
################################################################################

# Parse options
DO_PUSH=false
COMMIT_MESSAGE=""
QWEN_SCRIPT=""
OUTPUT_PATH=""
OUTPUT_PATH_NORMALIZED=""
NKILIB_MODE="bundled"
while [[ "$1" == --* ]]; do
    case "$1" in
        --push)
            DO_PUSH=true
            shift
            ;;
        --commit-message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        --qwen-script)
            QWEN_SCRIPT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --nkilib-mode)
            NKILIB_MODE="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a command to run on remote server"
    echo "Usage: ./remote_test.sh [--push] \"<command>\""
    echo ""
    echo "Options:"
    echo "  --push    Sync local changes to remote via git before running"
    echo ""
    echo "Examples:"
    echo "  ./remote_test.sh \"python test.py\""
    echo "  ./remote_test.sh --push \"python test.py\""
    echo "  ./remote_test.sh \"pytest -v tests/\""
    exit 1
fi

REMOTE_COMMAND="$1"
QWEN_MODULE=""

if [[ "$NKILIB_MODE" != "auto" && "$NKILIB_MODE" != "bundled" ]]; then
    echo "Error: --nkilib-mode must be one of: auto, bundled"
    exit 1
fi

if [ -n "$QWEN_SCRIPT" ]; then
    QWEN_MODULE="${QWEN_SCRIPT##*/}"
    QWEN_MODULE="${QWEN_MODULE%.py}"
fi

if [ -n "$OUTPUT_PATH" ]; then
    OUTPUT_PATH_NORMALIZED="${OUTPUT_PATH#./}"
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    exec > >(tee "$OUTPUT_PATH") 2>&1
    echo "[Local] Saving combined output to $OUTPUT_PATH"
fi

if [ -n "$QWEN_MODULE" ] && echo "$REMOTE_COMMAND" | grep -q "main.py" && ! echo "$REMOTE_COMMAND" | grep -Eq '(^|[[:space:]])--qwen([=[:space:]])'; then
    REMOTE_COMMAND="$REMOTE_COMMAND --qwen $QWEN_MODULE"
fi

echo "=========================================="
echo "Remote Test Script - NKI-MOE Project"
echo "=========================================="
echo ""

if [ "$DO_PUSH" = true ]; then
    CURRENT_BRANCH="$(git branch --show-current)"
    if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
        echo "Error: remote_test.sh expects current branch '$BRANCH', got '$CURRENT_BRANCH'"
        exit 1
    fi

    echo "[Sync 1/2] Staging and committing changes on $BRANCH branch..."
    git add -u
    while IFS= read -r untracked_file; do
        [ "$untracked_file" = ".codex" ] && continue
        [ "$untracked_file" = "$OUTPUT_PATH_NORMALIZED" ] && continue
        case "$untracked_file" in
            tmp/test_results/*) continue ;;
            results/*) continue ;;
        esac
        [ -z "$untracked_file" ] && continue
        git add -- "$untracked_file"
    done < <(git ls-files --others --exclude-standard)

    if git diff --cached --quiet; then
        echo "  ✓ No new staged changes to commit"
    else
        if [ -z "$COMMIT_MESSAGE" ]; then
            echo "Error: --commit-message is required when --push needs to create a commit"
            exit 1
        fi
        git commit -m "$COMMIT_MESSAGE"
        echo "  ✓ Changes committed"
    fi
    echo ""

    # Step 2: Push to remote
    echo "[Sync 2/2] Pushing to remote repository..."
    if git push origin "$BRANCH" 2>/dev/null; then
        echo "  ✓ Successfully pushed to origin/$BRANCH"
        SYNC_COMMAND="git checkout $BRANCH && git pull --ff-only origin $BRANCH"
    elif git push "$BRANCH" 2>/dev/null; then
        echo "  ✓ Successfully pushed to $BRANCH"
        SYNC_COMMAND="git checkout $BRANCH && git pull --ff-only origin $BRANCH"
    else
        echo "  ⚠ Push to origin failed, falling back to git bundle sync"
        BUNDLE_LOCAL_PATH="$(mktemp /tmp/nki_remote_test_XXXX.bundle)"
        git bundle create "$BUNDLE_LOCAL_PATH" "$BRANCH"
        ssh "$REMOTE_HOST" "cat > '$BUNDLE_REMOTE_PATH'" < "$BUNDLE_LOCAL_PATH"
        rm -f "$BUNDLE_LOCAL_PATH"
        echo "  ✓ Bundle transferred to $REMOTE_HOST:$BUNDLE_REMOTE_PATH"
        SYNC_COMMAND="git checkout $BRANCH && git pull --ff-only '$BUNDLE_REMOTE_PATH' $BRANCH && rm -f '$BUNDLE_REMOTE_PATH'"
    fi
    echo ""
else
    echo "[Skip] Git sync skipped (use --push to enable)"
    echo ""
fi

# Execute command on remote server
echo "[Run] Executing command on remote server..."
echo "  Remote host: $REMOTE_HOST"
echo "  Remote path: $REMOTE_PROJECT_DIR"
echo "  Command: $REMOTE_COMMAND"
echo ""
echo "--- Remote Output Start ---"

# Determine whether to sync code on remote
SYNC_COMMAND="${SYNC_COMMAND:-}"

# Base64-encode commands and config to safely pass them through SSH without quoting issues
SYNC_CMD_B64=$(printf '%s' "$SYNC_COMMAND" | base64)
REMOTE_CMD_B64=$(printf '%s' "$REMOTE_COMMAND" | base64)
PROJECT_DIR_B64=$(printf '%s' "$REMOTE_PROJECT_DIR" | base64)
VENV_B64=$(printf '%s' "$REMOTE_VENV" | base64)
MODEL_PATH_B64=$(printf '%s' "$REMOTE_MODEL_PATH" | base64)
COMPILED_MODEL_PATH_B64=$(printf '%s' "$REMOTE_COMPILED_MODEL_PATH" | base64)
NEURON_PLAT_B64=$(printf '%s' "$NEURON_PLATFORM" | base64)
NKILIB_MODE_B64=$(printf '%s' "$NKILIB_MODE" | base64)

# Disable set -e to capture exit code from SSH command
set +e
# Pass base64-encoded commands as env vars via the remote script header,
# then use a single-quoted heredoc so nothing is expanded locally.
# Use -tt to force PTY allocation for unbuffered output.
ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=40 "$REMOTE_HOST" "SYNC_CMD_B64='${SYNC_CMD_B64}' REMOTE_CMD_B64='${REMOTE_CMD_B64}' PROJECT_DIR_B64='${PROJECT_DIR_B64}' VENV_B64='${VENV_B64}' MODEL_PATH_B64='${MODEL_PATH_B64}' COMPILED_MODEL_PATH_B64='${COMPILED_MODEL_PATH_B64}' NEURON_PLAT_B64='${NEURON_PLAT_B64}' NKILIB_MODE_B64='${NKILIB_MODE_B64}' bash" <<'REMOTE_SCRIPT'
  set -e

  echo "[Remote Debug] Connected to remote server: $(hostname)"
  echo "[Remote Debug] Current time: $(date)"
  echo "[Remote Debug] Working directory: $(pwd)"

  # Decode config from base64
  PROJECT_DIR=$(echo "$PROJECT_DIR_B64" | base64 -d)
  VENV_PATH=$(echo "$VENV_B64" | base64 -d)
  MODEL_PATH=$(echo "$MODEL_PATH_B64" | base64 -d)
  COMPILED_MODEL_PATH=$(echo "$COMPILED_MODEL_PATH_B64" | base64 -d)
  NEURON_PLAT=$(echo "$NEURON_PLAT_B64" | base64 -d)
  NKILIB_MODE=$(echo "$NKILIB_MODE_B64" | base64 -d)

  cd "$PROJECT_DIR"
  echo "[Remote Debug] Changed to: $(pwd)"

  # Decode commands from base64
  SYNC_CMD=$(echo "$SYNC_CMD_B64" | base64 -d)
  REMOTE_CMD=$(echo "$REMOTE_CMD_B64" | base64 -d)
  echo "[Remote Debug] Decoded SYNC_CMD: '$SYNC_CMD'"
  echo "[Remote Debug] Decoded REMOTE_CMD: '$REMOTE_CMD'"

  # Sync code from remote repository (only if --push was used)
  if [ -n "$SYNC_CMD" ]; then
    echo "[Remote Debug] Running sync command..."
    eval "$SYNC_CMD"
    echo "[Remote Debug] Sync completed."
  else
    echo "[Remote Debug] No sync command, skipping."
  fi

  # Activate virtual environment
  echo "[Remote Debug] Activating virtual environment..."
  source "$VENV_PATH"
  echo "[Remote Debug] Python: $(which python) ($(python --version 2>&1))"

  # Set environment variables
  # Add project dir to PYTHONPATH but exclude nkilib_src to avoid
  # nkilib swap mechanism picking up incompatible nkilib_src (nki.tensor not available in nki 0.3.0)
  # Work around: nkilib_src requires nki.tensor which doesn't exist in remote nki 0.3.0
  export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
  export PYTHONUNBUFFERED=1
  export PROJECT_DIR=$PROJECT_DIR
  # Avoid forcing DGE notifications during baseline runs. On trn3 + Neuron 2.29
  # this can overflow the runtime notification queue during warmup/execution.
  unset NEURON_RT_ENABLE_DGE_NOTIFICATIONS
  export NEURON_PLATFORM_TARGET_OVERRIDE=$NEURON_PLAT
  # Default compile work dir and compile cache URL. Override on the command
  # line if a different location is needed.
  export BASE_COMPILE_WORK_DIR=/tmp/nxd_model_gsw/
  export NEURON_COMPILE_CACHE_URL=/tmp/nxd_compile_${USER}/
  if [ "$NKILIB_MODE" = "bundled" ]; then
    export NKILIB_FORCE_BUNDLED_LIBRARY=1
  else
    unset NKILIB_FORCE_BUNDLED_LIBRARY
  fi
  export MODEL_ARGS="--model-path $MODEL_PATH --compiled-model-path $COMPILED_MODEL_PATH"
  # export XLA_IR_DEBUG=1
  # export XLA_HLO_DEBUG=1

  echo "[Remote Debug] PYTHONPATH=$PYTHONPATH"
  echo "[Remote Debug] NKILIB_MODE=$NKILIB_MODE"
  echo "[Remote Debug] NKILIB_FORCE_BUNDLED_LIBRARY=${NKILIB_FORCE_BUNDLED_LIBRARY:-<unset>}"
  echo "[Remote Debug] Starting command execution..."
  echo "==========================================="

  # Execute the user-specified command
  eval "$REMOTE_CMD"
  EXIT_STATUS=$?
  exit $EXIT_STATUS
REMOTE_SCRIPT
EXIT_CODE=$?
set -e
echo "--- Remote Output End ---"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Remote command completed successfully (exit code: 0)"
else
    echo "✗ Remote command failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
