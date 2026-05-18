"""
Neuron Profiler Session Module.

Provides a context manager and CLI wrapper for profiling any NKI kernel.
The profiling infrastructure (env setup, report extraction, SCP) is decoupled
from the kernel code itself.

== As a Python module (context manager) ==

    from skills.profiler_analysis.profile_kernel import profiler_session

    with profiler_session("my_kernel") as session:
        import torch
        import torch_xla.core.xla_model as xm
        # ... your kernel code ...
        xm.mark_step()
        xm.wait_device_ops()

    # session.report_dir has the extracted reports

== As a CLI wrapper for an existing test file ==

    # Wrap any python script — sets profiling env, runs it, extracts reports
    python skills/profiler_analysis/profile_kernel.py --wrap "python ops/moe/test_moe_jit.py"

== Via remote_test.sh ==

    ./remote_test.sh --push "python skills/profiler_analysis/profile_kernel.py --wrap 'python ops/moe/test_moe_jit.py' "
"""

import argparse
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_WORK_DIR = os.path.join(PROJECT_ROOT, "tmp", "profiler_workspace")

_PROFILING_ENV_VARS = [
    "NEURON_RT_INSPECT_ENABLE",
    "NEURON_RT_INSPECT_OUTPUT_DIR",
    "NEURON_RT_INSPECT_DEVICE_PROFILE",
    "NEURON_RT_INSPECT_SYSTEM_PROFILE",
    "XLA_IR_DEBUG",
    "XLA_HLO_DEBUG",
    "NEURON_FRAMEWORK_DEBUG",
]


def _run_cmd(cmd, timeout=120):
    """Run a shell command and print output."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
        print(f"  stdout: {result.stdout[:2000]}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:2000]}")
    return result


def _find_files(root, ext):
    """Recursively find files by extension under root."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(ext):
                found.append(os.path.join(dirpath, f))
    return sorted(found, key=os.path.getmtime, reverse=True)


# ---------------------------------------------------------------------------
# Report extraction
# ---------------------------------------------------------------------------


def extract_device_reports(neff_path, ntff_path, report_dir):
    """Extract device-level reports from a NEFF+NTFF pair."""
    os.makedirs(report_dir, exist_ok=True)
    neff_name = os.path.splitext(os.path.basename(neff_path))[0]

    print()
    print("-" * 60)
    print(f"  Extracting device reports for: {os.path.basename(neff_path)}")
    print("-" * 60)

    # summary-json
    result = _run_cmd(
        [
            "neuron-profile",
            "view",
            "--output-format",
            "summary-json",
            "-n",
            neff_path,
            "-s",
            ntff_path,
        ],
        timeout=120,
    )
    if result.returncode == 0 and result.stdout.strip():
        path = os.path.join(report_dir, f"{neff_name}_summary.json")
        with open(path, "w") as f:
            f.write(result.stdout.strip())
        print(f"    -> {path}")

    # summary-text
    result = _run_cmd(
        [
            "neuron-profile",
            "view",
            "--output-format",
            "summary-text",
            "-n",
            neff_path,
            "-s",
            ntff_path,
        ],
        timeout=120,
    )
    if result.returncode == 0 and result.stdout.strip():
        path = os.path.join(report_dir, f"{neff_name}_summary.txt")
        with open(path, "w") as f:
            f.write(result.stdout)
        print(f"    -> {path}")

    # full json
    json_path = os.path.join(report_dir, f"{neff_name}_full.json")
    _run_cmd(
        [
            "neuron-profile",
            "view",
            "--output-format",
            "json",
            "--output-file",
            json_path,
            "-n",
            neff_path,
            "-s",
            ntff_path,
        ],
        timeout=120,
    )
    if os.path.exists(json_path):
        print(f"    -> {json_path} ({os.path.getsize(json_path):,} bytes)")

    # show-session
    result = _run_cmd(
        [
            "neuron-profile",
            "show-session",
            "-s",
            ntff_path,
            "-j",
        ],
        timeout=60,
    )
    if result.returncode == 0 and result.stdout.strip():
        path = os.path.join(report_dir, f"{neff_name}_session.json")
        with open(path, "w") as f:
            f.write(result.stdout)
        print(f"    -> {path}")


def extract_system_reports(output_dir, report_dir):
    """Extract system-level reports from inspect output."""
    os.makedirs(report_dir, exist_ok=True)

    print()
    print("-" * 60)
    print("  Extracting system-level reports ...")
    print("-" * 60)

    result = _run_cmd(
        [
            "neuron-profile",
            "view",
            "-d",
            output_dir,
            "--output-format",
            "summary-json",
        ],
        timeout=120,
    )
    if result.returncode == 0 and result.stdout.strip():
        path = os.path.join(report_dir, "system_summary.json")
        with open(path, "w") as f:
            f.write(result.stdout.strip())
        print(f"    -> {path}")

    result = _run_cmd(
        [
            "neuron-profile",
            "view",
            "-d",
            output_dir,
            "--output-format",
            "summary-text",
        ],
        timeout=120,
    )
    if result.returncode == 0 and result.stdout.strip():
        path = os.path.join(report_dir, "system_summary.txt")
        with open(path, "w") as f:
            f.write(result.stdout)
        print(f"    -> {path}")

    json_path = os.path.join(report_dir, "system_full.json")
    _run_cmd(
        [
            "neuron-profile",
            "view",
            "-d",
            output_dir,
            "--output-format",
            "json",
            "--output-file",
            json_path,
        ],
        timeout=120,
    )
    if os.path.exists(json_path):
        print(f"    -> {json_path} ({os.path.getsize(json_path):,} bytes)")


_END_TO_END_THRESHOLD = 2  # > this many NEFFs = end-to-end mode (use median size)


def _extract_neff_id(path):
    """Extract the numeric ID from a NEFF or NTFF filename.

    NEFF naming: neff_{ID}_vnc_{core}.neff  OR  neff_{ID}.neff
    NTFF naming: {ID}_vnc_{core}.ntff
    """
    basename = os.path.basename(path)
    if basename.endswith(".neff"):
        # Strip prefix and extension: neff_{ID}_vnc_{core}.neff -> {ID}_vnc_{core}
        stem = basename.replace("neff_", "").replace(".neff", "")
        # Strip _vnc_{core} suffix if present
        parts = stem.split("_vnc_")
        return parts[0] if parts else stem
    elif basename.endswith(".ntff"):
        # {ID}_vnc_{core}.ntff
        parts = basename.split("_vnc_")
        if parts:
            return parts[0]
    return None


def _select_neff(files):
    """
    Auto-select the best NEFF from a list based on file count.

    - Single kernel mode (<=4 files): pick the largest — the target kernel
      is almost always the biggest NEFF.
    - End-to-end mode (>4 files): pick median by file size — extremes are
      H2D transfers or oversized fused graphs.
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    sorted_files = sorted(files, key=lambda f: os.path.getsize(f))
    if len(files) <= _END_TO_END_THRESHOLD:
        # Single kernel: largest
        return sorted_files[-1]
    else:
        # End-to-end: median
        return sorted_files[len(sorted_files) // 2]


def _select_neff_ntff_pair(neffs, ntffs):
    """Select a matching NEFF + NTFF pair by ID.

    NEFF and NTFF files share numeric IDs:
      - NEFF: neff_{ID}.neff
      - NTFF: {ID}_vnc_{core}.ntff  (one per neuron core)

    Strategy:
      1. Select the best NEFF via _select_neff (largest or median).
      2. Find NTFFs with the same ID.
      3. Pick the largest matching NTFF (most data captured).
    """
    selected_neff = _select_neff(neffs)
    if selected_neff is None:
        return None, None

    neff_id = _extract_neff_id(selected_neff)

    # Find NTFFs with matching ID
    matching_ntffs = [f for f in ntffs if _extract_neff_id(f) == neff_id]
    if not matching_ntffs:
        # Fallback: return largest NTFF with warning
        print(
            f"  WARNING: No NTFF found matching NEFF ID {neff_id}, falling back to largest NTFF"
        )
        matching_ntffs = ntffs

    # Pick the largest matching NTFF
    selected_ntff = max(matching_ntffs, key=lambda f: os.path.getsize(f))
    return selected_neff, selected_ntff


def extract_all_reports(inspect_output_dir, report_dir):
    """Extract both device and system reports from an inspect output directory."""
    neffs = _find_files(inspect_output_dir, ".neff")
    ntffs = _find_files(inspect_output_dir, ".ntff")

    if neffs and ntffs:
        mode = "end-to-end" if len(neffs) > _END_TO_END_THRESHOLD else "single-kernel"
        print(f"  Mode: {mode} ({len(neffs)} NEFFs, {len(ntffs)} NTFFs)")

        selected_neff, selected_ntff = _select_neff_ntff_pair(neffs, ntffs)
        neff_id = _extract_neff_id(selected_neff)
        ntff_id = _extract_neff_id(selected_ntff)
        strategy = "median by size" if mode == "end-to-end" else "largest"
        print(f"  Strategy: {strategy}")
        print(
            f"  Selected NEFF: {os.path.basename(selected_neff)} (ID={neff_id}, {os.path.getsize(selected_neff):,} bytes)"
        )
        print(
            f"  Selected NTFF: {os.path.basename(selected_ntff)} (ID={ntff_id}, {os.path.getsize(selected_ntff):,} bytes)"
        )
        extract_device_reports(selected_neff, selected_ntff, report_dir)
    else:
        print("  WARNING: No NEFF/NTFF found. Profile may not have been captured.")
        if not neffs:
            print("    Missing: .neff files")
        if not ntffs:
            print("    Missing: .ntff files")

    if os.path.isdir(inspect_output_dir):
        extract_system_reports(inspect_output_dir, report_dir)


# ---------------------------------------------------------------------------
# Profiler Session (context manager)
# ---------------------------------------------------------------------------


class ProfilerSession:
    """
    Context manager that sets up Neuron profiling env vars on enter,
    and extracts reports on exit.

    Usage:
        with profiler_session("my_kernel") as s:
            # your kernel code here (import torch, run kernel, etc.)
            pass
        print(s.report_dir)  # path to extracted reports
    """

    def __init__(self, name="kernel", work_dir=None):
        self.name = name
        self.work_dir = work_dir or DEFAULT_WORK_DIR
        self.inspect_output_dir = os.path.join(self.work_dir, "inspect_output")
        self.report_dir = os.path.join(self.work_dir, "reports")
        self._saved_env = {}

    def __enter__(self):
        # Clean workspace
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.inspect_output_dir, exist_ok=True)

        # Save current env and set profiling vars
        env_settings = {
            "NEURON_RT_INSPECT_ENABLE": "1",
            "NEURON_RT_INSPECT_OUTPUT_DIR": self.inspect_output_dir,
            "NEURON_RT_INSPECT_DEVICE_PROFILE": "1",
            "NEURON_RT_INSPECT_SYSTEM_PROFILE": "1",
            "NEURON_RT_ENABLE_DGE_NOTIFICATIONS": "1",
            "XLA_IR_DEBUG": "1",
            "XLA_HLO_DEBUG": "1",
        }
        for k, v in env_settings.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v

        print("=" * 60)
        print(f"Profiler session started: {self.name}")
        print(f"  Output: {self.inspect_output_dir}")
        print("  Env vars set:")
        for k, v in env_settings.items():
            print(f"    {k}={v}")
        print("=" * 60)
        print()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore env
        for k, saved in self._saved_env.items():
            if saved is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved

        if exc_type is not None:
            print(f"\n  Kernel execution failed: {exc_type.__name__}: {exc_val}")
            return False

        # Extract reports
        print()
        print("=" * 60)
        print("Extracting reports ...")
        print("=" * 60)
        extract_all_reports(self.inspect_output_dir, self.report_dir)
        self._print_file_listing()

        return False

    def _print_file_listing(self):
        print()
        print("Files generated:")
        for root, dirs, files in os.walk(self.work_dir):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, self.work_dir)
                print(f"  {rel:60s} {os.path.getsize(fpath):>10,} bytes")


def profiler_session(name="kernel", work_dir=None):
    """Convenience function to create a ProfilerSession."""
    return ProfilerSession(name=name, work_dir=work_dir)


# ---------------------------------------------------------------------------
# CLI: --wrap mode
# ---------------------------------------------------------------------------


def _wrap_command(cmd_str, work_dir, view_port=None):
    """
    Run an arbitrary command as a subprocess with profiling env vars set.
    Then extract reports from the output.
    """
    inspect_output_dir = os.path.join(work_dir, "inspect_output")
    report_dir = os.path.join(work_dir, "reports")

    # Clean workspace
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(inspect_output_dir, exist_ok=True)

    # Build env with profiling vars
    env = os.environ.copy()
    env.update(
        {
            "NEURON_RT_INSPECT_ENABLE": "1",
            "NEURON_RT_INSPECT_OUTPUT_DIR": inspect_output_dir,
            "NEURON_RT_INSPECT_DEVICE_PROFILE": "1",
            "NEURON_RT_INSPECT_SYSTEM_PROFILE": "1",
            "XLA_IR_DEBUG": "1",
            "XLA_HLO_DEBUG": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    print("=" * 60)
    print(f"Wrapping command with profiling")
    print(f"  Command: {cmd_str}")
    print(f"  Output:  {inspect_output_dir}")
    print("=" * 60)
    print()

    # Run the command as subprocess
    result = subprocess.run(
        cmd_str,
        shell=True,
        env=env,
        timeout=3600,
    )

    if result.returncode != 0:
        print(f"\n  Command exited with code {result.returncode}")

    # List raw profile files sorted by size
    print()
    print("=" * 60)
    print("Profile data captured:")
    print("=" * 60)
    neffs = _find_files(inspect_output_dir, ".neff")
    ntffs = _find_files(inspect_output_dir, ".ntff")

    # Sort by size for display (ascending)
    neffs_by_size = sorted(neffs, key=lambda f: os.path.getsize(f))
    ntffs_by_size = sorted(ntffs, key=lambda f: os.path.getsize(f))

    print(f"  NEFFs ({len(neffs)}):")
    for f in neffs_by_size:
        print(f"    {os.path.basename(f):40s} {os.path.getsize(f):>12,} bytes")
    print(f"  NTFFs ({len(ntffs)}):")
    for f in ntffs_by_size:
        print(f"    {os.path.basename(f):40s} {os.path.getsize(f):>12,} bytes")

    # Always print the recommended selection
    if neffs and ntffs:
        selected_neff, selected_ntff = _select_neff_ntff_pair(neffs, ntffs)
        mode = "end-to-end" if len(neffs) > _END_TO_END_THRESHOLD else "single-kernel"
        strategy = "median by size" if mode == "end-to-end" else "largest"
        neff_id = _extract_neff_id(selected_neff)
        ntff_id = _extract_neff_id(selected_ntff)
        print()
        print(f"  Recommended ({mode}, strategy: {strategy}):")
        print(f"    NEFF: {selected_neff} (ID={neff_id})")
        print(f"    NTFF: {selected_ntff} (ID={ntff_id})")

    print(f"\nProfile data at: {inspect_output_dir}")

    # Extract reports (summary-json, summary-text, etc.) from the collected data
    if neffs and ntffs:
        print()
        extract_all_reports(inspect_output_dir, report_dir)

    # Auto-launch viewer if requested
    if view_port:
        if neffs and ntffs:
            _launch_viewer(selected_neff, selected_ntff, view_port)
        else:
            print("  Cannot launch viewer: no NEFF/NTFF found")

    return result.returncode


# ---------------------------------------------------------------------------
# Viewer launcher
# ---------------------------------------------------------------------------

VIEW_PORT_DEFAULT = 3001


def _launch_viewer(neff_path, ntff_path, port):
    """Kill existing neuron-profile viewer and start a new one."""
    print()
    print("=" * 60)
    print(f"Launching neuron-profile viewer on port {port}...")
    print("=" * 60)

    # Kill any existing viewer
    subprocess.run(["pkill", "-f", "neuron-profile"], capture_output=True)

    # Start viewer in background
    log_path = "/tmp/np_view.log"
    cmd = [
        "neuron-profile",
        "view",
        "-n",
        neff_path,
        "-s",
        ntff_path,
        "-p",
        str(port),
        "--force",
    ]
    with open(log_path, "w") as log_f:
        subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
        )

    # Wait for startup and print URL
    import time

    time.sleep(3)
    try:
        with open(log_path) as f:
            log_content = f.read()
        print(f"  {log_content.strip()}")
        # Extract URL from log
        for line in log_content.splitlines():
            if "View profile at" in line:
                print(f"\n  >>> Open in browser: {line.split('at ')[-1].strip()}")
                break
        else:
            print(f"\n  >>> Open in browser: http://localhost:{port}")
    except Exception:
        print(f"\n  >>> Open in browser: http://localhost:{port}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Neuron Profiler Session — wrap any command or test with profiling"
    )
    parser.add_argument(
        "--wrap",
        required=True,
        help='Command to run with profiling, e.g. "python ops/moe/test_moe_jit.py"',
    )
    parser.add_argument(
        "--workdir",
        default=DEFAULT_WORK_DIR,
        help=f"Working directory (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="(disabled) analyze_profile.py is not currently supported",
    )
    parser.add_argument(
        "--view",
        nargs="?",
        const=VIEW_PORT_DEFAULT,
        type=int,
        default=None,
        help=f"Launch neuron-profile viewer after profiling (default port: {VIEW_PORT_DEFAULT})",
    )
    args = parser.parse_args()

    if args.analyze:
        print("ERROR: --analyze is not currently supported.")
        sys.exit(1)

    exit_code = _wrap_command(args.wrap, args.workdir, view_port=args.view)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
