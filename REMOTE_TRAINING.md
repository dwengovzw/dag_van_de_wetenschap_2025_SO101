# Remote Training Server – Deployment Guide

This guide explains how to set up the remote training server on a GPU machine so the local data collection app can offload policy training.

## Prerequisites

- Python 3.10+
- A machine with a CUDA-capable GPU (recommended) or CPU (slower)
- Network access from the local machine to the server (port 5000 by default)

## 1. Clone the repository

```bash
git clone https://github.com/dwengovzw/dag_van_de_wetenschap_2025_SO101.git
cd dag_van_de_wetenschap_2025_SO101
```

## 2. Create a Python environment

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot
```

## 3. Install dependencies

```bash
pip install -r requirements-server.txt
```

This installs LeRobot (which includes `lerobot-train`) and Flask.

> **Note:** LeRobot will pull in PyTorch. If you need a specific CUDA version, install PyTorch first following [pytorch.org](https://pytorch.org/get-started/locally/) before running the pip install.

## 4. Choose an API key

Pick a secret key that both the server and the local app will share:

```bash
export TRAINING_API_KEY="your-secret-key-here"
```

## 5. Start the server

```bash
python remote_training_server.py --api-key "$TRAINING_API_KEY"
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Interface to bind to |
| `--port` | `5000` | Port to listen on |
| `--api-key` | (env var) | API key (or set `TRAINING_API_KEY` env var) |
| `--work-dir` | `./training_jobs` | Directory where job data is stored |
| `--train-bin` | `lerobot-train` | Path to the `lerobot-train` binary |

The server stores all datasets, training outputs, and job metadata under the work directory.

## 6. Configure the local app

On the machine running the data collection GUI, edit `tasks_config.json`:

```json
{
    "training_mode": "remote",
    "remote_server_url": "http://<server-ip>:5000",
    "remote_api_key": "your-secret-key-here",
    ...
}
```

Set `training_mode` to `"remote"` and fill in the server URL and matching API key. The local app will then upload datasets to the server for training and poll for the trained policy automatically.

## 7. Verify the connection

From the local machine, test that the server is reachable:

```bash
curl -H "X-API-Key: your-secret-key-here" http://<server-ip>:5000/jobs
```

You should get `{"jobs": []}`.

## How it works

1. The local app collects the required number of episodes for a task.
2. It tars and uploads the dataset to the server via `POST /jobs`.
3. The server extracts the dataset and runs `lerobot-train` as a subprocess.
4. The local app polls `GET /jobs/<id>` every 30 seconds until training completes.
5. Once done, the local app downloads the trained policy via `GET /jobs/<id>/policy` and extracts it locally.
6. The policy is then available for execution through the GUI.

If the local app is restarted during training, it automatically resumes polling for any in-progress remote jobs.

## Running as a systemd service (optional)

To keep the server running after logout:

```bash
sudo tee /etc/systemd/system/lerobot-training.service << EOF
[Unit]
Description=LeRobot Remote Training Server
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment="TRAINING_API_KEY=your-secret-key-here"
ExecStart=$(which python) remote_training_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now lerobot-training
```

Check status with `sudo systemctl status lerobot-training` and logs with `journalctl -u lerobot-training -f`.

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | `POST` | Upload dataset and start training (multipart form: `dataset` file + `task_name`, `policy_type`, `training_steps`, `batch_size` fields) |
| `/jobs` | `GET` | List all jobs |
| `/jobs/<id>` | `GET` | Get job status (`training`, `completed`, or `failed`) |
| `/jobs/<id>/policy` | `GET` | Download trained policy tarball (only when `completed`) |

All endpoints require the `X-API-Key` header.
