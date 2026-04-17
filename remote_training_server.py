"""
Remote Training Server for SO-101 Multi-Task System

Listens for training requests from the local data collection app,
runs lerobot-train on a GPU machine, and serves trained policies back.

Requirements:
    pip install flask

Usage:
    python remote_training_server.py
    python remote_training_server.py --port 5000 --host 0.0.0.0
    python remote_training_server.py --api-key my-secret-key
    TRAINING_API_KEY=my-secret-key python remote_training_server.py

The server stores all job data under ./training_jobs/ by default.
Each job gets its own directory with the uploaded dataset, training
output, and a metadata JSON file.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ===== Configuration =====
WORK_DIR = Path("./training_jobs")
JOBS_INDEX = WORK_DIR / "jobs.json"
TRAIN_BIN = "lerobot-train"  # Assumes lerobot-train is on PATH
API_KEY = os.environ.get("TRAINING_API_KEY", "change-me-to-a-secret")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

# In-memory job registry, persisted to disk
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ===== Persistence =====

def _load_jobs():
    """Load job registry from disk."""
    global _jobs
    if JOBS_INDEX.exists():
        with open(JOBS_INDEX) as f:
            _jobs = json.load(f)


def _save_jobs():
    """Persist job registry to disk."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOBS_INDEX, "w") as f:
        json.dump(_jobs, f, indent=2)


def _update_job(job_id: str, **updates):
    """Thread-safe job status update."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)
            _save_jobs()


# ===== Auth =====

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key != API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ===== Tar safety =====

def _safe_tar_extract(tar: tarfile.TarFile, dest: Path):
    """Extract tar safely, preventing path traversal attacks."""
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise ValueError(f"Path traversal detected in tar member: {member.name}")
    tar.extractall(dest)


# ===== Endpoints =====

@app.route("/jobs", methods=["POST"])
@require_api_key
def create_job():
    """Accept a dataset upload and start a training job."""
    if "dataset" not in request.files:
        return jsonify({"error": "No dataset file uploaded"}), 400

    task_name = request.form.get("task_name", "")
    if not task_name:
        return jsonify({"error": "task_name is required"}), 400

    policy_type = request.form.get("policy_type", "act")
    training_steps = int(request.form.get("training_steps", 100000))
    batch_size = int(request.form.get("batch_size", 8))

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save uploaded dataset tarball
    dataset_file = request.files["dataset"]
    tar_path = job_dir / "dataset.tar.gz"
    dataset_file.save(str(tar_path))

    # Extract dataset
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_tar_extract(tar, job_dir)
    except Exception as e:
        return jsonify({"error": f"Failed to extract dataset: {e}"}), 400

    config = {
        "task_name": task_name,
        "policy_type": policy_type,
        "training_steps": training_steps,
        "batch_size": batch_size,
    }

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "training",
            "config": config,
            "created_at": time.time(),
            "error": None,
        }
        _save_jobs()

    # Start training in background
    thread = threading.Thread(target=_train, args=(job_id,), daemon=True)
    thread.start()

    print(f"[JOB {job_id}] Training started for: {task_name}")
    return jsonify({"job_id": job_id, "status": "training"})


@app.route("/jobs/<job_id>", methods=["GET"])
@require_api_key
def get_job_status(job_id):
    """Check the status of a training job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "config": job.get("config"),
        "error": job.get("error"),
    })


@app.route("/jobs/<job_id>/log", methods=["GET"])
@require_api_key
def get_job_log(job_id):
    """Download the full training log for a job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    log_file = WORK_DIR / job_id / "train.log"
    if not log_file.exists():
        return jsonify({"error": "No log file available yet"}), 404

    return send_file(str(log_file), mimetype="text/plain")


@app.route("/jobs/<job_id>/policy", methods=["GET"])
@require_api_key
def download_policy(job_id):
    """Download the trained policy for a completed job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Training not yet completed"}), 404

    policy_tar = WORK_DIR / job_id / "policy.tar.gz"
    if not policy_tar.exists():
        return jsonify({"error": "Policy archive not found"}), 404

    return send_file(str(policy_tar), mimetype="application/gzip")


@app.route("/jobs", methods=["GET"])
@require_api_key
def list_jobs():
    """List all training jobs."""
    with _jobs_lock:
        summary = []
        for jid, job in _jobs.items():
            summary.append({
                "job_id": jid,
                "status": job["status"],
                "task_name": job.get("config", {}).get("task_name", ""),
            })
    return jsonify({"jobs": summary})


# ===== Training =====

def _train(job_id: str):
    """Run lerobot-train as a subprocess."""
    job_dir = WORK_DIR / job_id

    with _jobs_lock:
        config = _jobs[job_id]["config"]

    dataset_dir = job_dir / "dataset"
    output_dir = job_dir / "output"
    safe_name = config["task_name"].lower().replace(" ", "_").replace("/", "_")
    repo_id = f"local/{safe_name}"

    cmd = [
        TRAIN_BIN,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_dir}",
        f"--policy.type={config['policy_type']}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--steps={config['training_steps']}",
        f"--batch_size={config['batch_size']}",
    ]

    log_file = job_dir / "train.log"
    print(f"[JOB {job_id}] Running: {' '.join(cmd)}")

    try:
        with open(log_file, "w") as lf:
            lf.write(f"Command: {' '.join(cmd)}\n\n")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)

        with open(log_file, "a") as lf:
            lf.write(f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}\n")
            lf.write(f"\nReturn code: {result.returncode}\n")

        if result.returncode == 0:
            # Package the training output as a tarball for download
            policy_tar = job_dir / "policy.tar.gz"
            with tarfile.open(policy_tar, "w:gz") as tar:
                tar.add(str(output_dir), arcname=".")

            _update_job(job_id, status="completed")
            print(f"[JOB {job_id}] Training completed successfully")
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            _update_job(job_id, status="failed", error=error_msg)
            print(f"[JOB {job_id}] Training failed: {error_msg[:200]}")

    except subprocess.TimeoutExpired:
        _update_job(job_id, status="failed", error="Training timed out (24h limit)")
        print(f"[JOB {job_id}] Training timed out")
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e))
        print(f"[JOB {job_id}] Training error: {e}")


# ===== Startup =====

def _resume_incomplete_jobs():
    """On server restart, mark any 'training' jobs as failed (no subprocess to resume)."""
    with _jobs_lock:
        for job_id, job in _jobs.items():
            if job["status"] == "training":
                job["status"] = "failed"
                job["error"] = "Server restarted while training was in progress"
                print(f"[JOB {job_id}] Marked as failed (server restart)")
        _save_jobs()


def main():
    parser = argparse.ArgumentParser(description="Remote training server for SO-101 multi-task system")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--api-key", default=None, help="API key for authentication (or set TRAINING_API_KEY env var)")
    parser.add_argument("--work-dir", default="./training_jobs", help="Directory for job data (default: ./training_jobs)")
    parser.add_argument("--train-bin", default=None, help="Path to lerobot-train binary (default: lerobot-train on PATH)")
    args = parser.parse_args()

    global API_KEY, WORK_DIR, JOBS_INDEX, TRAIN_BIN

    if args.api_key:
        API_KEY = args.api_key
    if args.work_dir:
        WORK_DIR = Path(args.work_dir)
        JOBS_INDEX = WORK_DIR / "jobs.json"
    if args.train_bin:
        TRAIN_BIN = args.train_bin

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    _load_jobs()
    _resume_incomplete_jobs()

    if API_KEY == "change-me-to-a-secret":
        print("WARNING: Using default API key. Set --api-key or TRAINING_API_KEY env var for security.")

    print(f"Starting remote training server on {args.host}:{args.port}")
    print(f"Work directory: {WORK_DIR.resolve()}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
