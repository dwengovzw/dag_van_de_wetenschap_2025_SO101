"""
Multi-Task Faire Booth GUI for SO-101 Robot

A tkinter-based GUI supporting multiple tasks, data collection,
background policy training, and policy execution.

Screens:
    task_overview  - Shows all tasks, their progress, and available actions
    idle           - Waiting to start an episode for the selected task
    recording      - Episode being recorded (teleop)
    review         - User decides if episode was good
    timeout        - Episode ran out of time
    saving         - Brief feedback while saving
    policy_running - Policy is executing on the robot
    done           - All tasks completed
"""

import tkinter as tk
from tkinter import font as tkfont


COLORS = {
    "bg": "#1a1a2e",
    "card": "#16213e",
    "accent": "#0f3460",
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "text": "#ecf0f1",
    "text_dim": "#95a5a6",
    "button_start": "#27ae60",
    "button_stop": "#e67e22",
    "button_yes": "#2ecc71",
    "button_no": "#e74c3c",
    "button_quit": "#c0392b",
    "button_policy": "#8e44ad",
    "button_back": "#7f8c8d",
    "training": "#3498db",
}


class FaireMultiTaskGUI:
    """
    Multi-task GUI for faire booth data collection and policy execution.

    The GUI communicates with the main script through a shared `events` dict
    and a `task_state` dict that tracks per-task progress.
    """

    def __init__(self, events: dict, task_configs: list[dict], task_state: dict):
        """
        Args:
            events: Shared event dict for controlling recording/policy threads.
            task_configs: List of task config dicts from tasks_config.json.
            task_state: Shared dict tracking per-task state:
                {task_name: {"episodes_collected": int, "training_status": str, "policy_path": str|None}}
                training_status: "not_started", "training", "trained", "failed"
        """
        self.events = events
        self.task_configs = task_configs
        self.task_state = task_state
        self.current_task = None  # Index into task_configs
        self.state = "task_overview"
        self._timer_id = None
        self._overview_refresh_id = None
        self._remaining_s = 0
        self.task_list_frame = None
        self.recording_tip_label = None
        self._card_widgets = []  # Per-card mutable widget references
        self._log_text = None  # Text widget for training log viewer
        self._log_refresh_id = None
        self._log_task_name = None

        self._build_ui()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Robot Multi-Task Data Collection")
        self.root.configure(bg=COLORS["bg"])
        self.root.attributes("-zoomed", True)
        self.root.minsize(900, 700)

        self.font_title = tkfont.Font(family="Helvetica", size=36, weight="bold")
        self.font_large = tkfont.Font(family="Helvetica", size=24)
        self.font_medium = tkfont.Font(family="Helvetica", size=18)
        self.font_button = tkfont.Font(family="Helvetica", size=20, weight="bold")
        self.font_small = tkfont.Font(family="Helvetica", size=14)
        self.font_task_title = tkfont.Font(family="Helvetica", size=16, weight="bold")
        self.font_task_detail = tkfont.Font(family="Helvetica", size=13)

        # Main container
        self.container = tk.Frame(self.root, bg=COLORS["bg"])
        self.container.pack(expand=True, fill="both", padx=40, pady=30)

        # Title
        tk.Label(
            self.container, text="🤖 Robot Multi-Task Collection",
            font=self.font_title, fg=COLORS["text"], bg=COLORS["bg"]
        ).pack(pady=(0, 10))

        # Subtitle / breadcrumb
        self.subtitle_label = tk.Label(
            self.container, text="", font=self.font_medium,
            fg=COLORS["text_dim"], bg=COLORS["bg"]
        )
        self.subtitle_label.pack(pady=(0, 15))

        # Content area (swapped per screen)
        self.content_frame = tk.Frame(self.container, bg=COLORS["bg"])
        self.content_frame.pack(expand=True, fill="both")

        # Bottom bar with quit button
        bottom = tk.Frame(self.container, bg=COLORS["bg"])
        bottom.pack(fill="x", pady=(10, 0))
        tk.Button(
            bottom, text="Exit", font=self.font_small,
            bg=COLORS["button_quit"], fg=COLORS["text"],
            activebackground="#a93226", activeforeground=COLORS["text"],
            relief="flat", padx=20, pady=8, command=self._on_quit
        ).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self._show_task_overview()

    def _clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def _make_button(self, parent, text, color, command, font=None):
        btn = tk.Button(
            parent, text=text, font=font or self.font_button,
            bg=color, fg=COLORS["text"],
            activebackground=color, activeforeground=COLORS["text"],
            relief="flat", padx=30, pady=15, command=command
        )
        return btn

    # ================================================================
    # Task Overview Screen
    # ================================================================

    def _show_task_overview(self):
        self.state = "task_overview"
        self.current_task = None
        self._cancel_timer()
        self._clear_content()
        self.subtitle_label.config(text="Select a task to collect data or run a trained policy")

        # Scrollable task list
        canvas = tk.Canvas(self.content_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        self.task_list_frame = tk.Frame(canvas, bg=COLORS["bg"])

        self.task_list_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.task_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._populate_task_cards()
        self._schedule_overview_refresh()

    def _populate_task_cards(self):
        """Build task cards once. Stores references to mutable widgets in _card_widgets."""
        for widget in self.task_list_frame.winfo_children():
            widget.destroy()
        self._card_widgets = []

        for idx, task in enumerate(self.task_configs):
            name = task["name"]
            state = self.task_state.get(name, {})
            collected = state.get("episodes_collected", 0)
            required = task["required_episodes"]
            training_status = state.get("training_status", "not_started")
            policy_path = state.get("policy_path")

            # Card
            card = tk.Frame(self.task_list_frame, bg=COLORS["card"], padx=20, pady=15)
            card.pack(fill="x", pady=8, padx=10)

            # Top row: task name + status
            top = tk.Frame(card, bg=COLORS["card"])
            top.pack(fill="x")

            tk.Label(
                top, text=name, font=self.font_task_title,
                fg=COLORS["text"], bg=COLORS["card"], anchor="w"
            ).pack(side="left")

            # Training status badge (mutable)
            badge_text, badge_color = self._training_badge(training_status)
            badge_label = tk.Label(
                top, text=badge_text, font=self.font_task_detail,
                fg=badge_color, bg=COLORS["card"]
            )
            badge_label.pack(side="right", padx=(10, 0))

            # Description
            tk.Label(
                card, text=task.get("description", ""),
                font=self.font_task_detail, fg=COLORS["text_dim"],
                bg=COLORS["card"], anchor="w", wraplength=600
            ).pack(fill="x", pady=(5, 5))

            # Progress bar (mutable)
            progress_frame = tk.Frame(card, bg=COLORS["card"])
            progress_frame.pack(fill="x", pady=(5, 5))

            bar_bg = tk.Frame(progress_frame, bg="#34495e", height=20)
            bar_bg.pack(fill="x")
            bar_bg.pack_propagate(False)

            frac = min(collected / required, 1.0) if required > 0 else 0
            bar_color = COLORS["success"] if frac >= 1.0 else COLORS["warning"]
            bar_fill = tk.Frame(bar_bg, bg=bar_color)
            if frac > 0:
                bar_fill.place(relx=0, rely=0, relwidth=frac, relheight=1.0)

            progress_label = tk.Label(
                progress_frame,
                text=f"{collected} / {required} episodes",
                font=self.font_task_detail, fg=COLORS["text_dim"], bg=COLORS["card"]
            )
            progress_label.pack(pady=(3, 0))

            # Buttons row (mutable — policy button may appear/disappear)
            btn_row = tk.Frame(card, bg=COLORS["card"])
            btn_row.pack(fill="x", pady=(8, 0))

            collect_btn = self._make_button(
                btn_row, "📋 Collect Data", COLORS["button_start"],
                lambda i=idx: self._select_task(i, mode="collect"),
                font=self.font_small
            )
            collect_btn.pack(side="left", padx=(0, 10))

            policy_btn = None
            if policy_path and training_status == "trained":
                policy_btn = self._make_button(
                    btn_row, "🤖 Run Policy", COLORS["button_policy"],
                    lambda i=idx: self._select_task(i, mode="policy"),
                    font=self.font_small
                )
                policy_btn.pack(side="left", padx=(0, 10))

            # Train button — shown when not currently training
            train_btn = None
            if training_status not in ("training",):
                train_btn = self._make_button(
                    btn_row, "🧠 Train", COLORS["training"],
                    lambda i=idx: self._on_start_training(i),
                    font=self.font_small
                )
                train_btn.pack(side="left", padx=(0, 10))

            # View log button — shown when a log exists
            log_btn = None
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            log_exists = (self._get_data_dir() / safe_name / "train.log").exists()
            if log_exists or training_status in ("training", "trained", "failed"):
                log_btn = self._make_button(
                    btn_row, "📄 Log", COLORS["accent"],
                    lambda i=idx: self._show_training_log(i),
                    font=self.font_small
                )
                log_btn.pack(side="left", padx=(0, 10))

            self._card_widgets.append({
                "badge_label": badge_label,
                "bar_bg": bar_bg,
                "bar_fill": bar_fill,
                "progress_label": progress_label,
                "btn_row": btn_row,
                "policy_btn": policy_btn,
                "train_btn": train_btn,
                "log_btn": log_btn,
                "idx": idx,
            })

    def _training_badge(self, status):
        badges = {
            "not_started": ("○ Not trained", COLORS["text_dim"]),
            "training": ("⟳ Training...", COLORS["training"]),
            "trained": ("✓ Policy ready", COLORS["success"]),
            "failed": ("✗ Training failed", COLORS["danger"]),
        }
        return badges.get(status, ("?", COLORS["text_dim"]))

    def _schedule_overview_refresh(self):
        """Periodically update dynamic parts of task cards without rebuilding."""
        if self.state != "task_overview":
            return

        for cw in self._card_widgets:
            idx = cw["idx"]
            task = self.task_configs[idx]
            name = task["name"]
            state = self.task_state.get(name, {})
            collected = state.get("episodes_collected", 0)
            required = task["required_episodes"]
            training_status = state.get("training_status", "not_started")
            policy_path = state.get("policy_path")

            # Update badge
            badge_text, badge_color = self._training_badge(training_status)
            cw["badge_label"].config(text=badge_text, fg=badge_color)

            # Update progress bar
            frac = min(collected / required, 1.0) if required > 0 else 0
            bar_color = COLORS["success"] if frac >= 1.0 else COLORS["warning"]
            cw["bar_fill"].config(bg=bar_color)
            if frac > 0:
                cw["bar_fill"].place(relx=0, rely=0, relwidth=frac, relheight=1.0)
            else:
                cw["bar_fill"].place_forget()

            # Update episode count
            cw["progress_label"].config(text=f"{collected} / {required} episodes")

            # Show/hide policy button
            if policy_path and training_status == "trained":
                if cw["policy_btn"] is None:
                    cw["policy_btn"] = self._make_button(
                        cw["btn_row"], "🤖 Run Policy", COLORS["button_policy"],
                        lambda i=idx: self._select_task(i, mode="policy"),
                        font=self.font_small
                    )
                    cw["policy_btn"].pack(side="left", padx=(0, 10))
            else:
                if cw["policy_btn"] is not None:
                    cw["policy_btn"].destroy()
                    cw["policy_btn"] = None

            # Show/hide train button (hide while training is in progress)
            if training_status in ("training",):
                if cw["train_btn"] is not None:
                    cw["train_btn"].destroy()
                    cw["train_btn"] = None
            else:
                if cw["train_btn"] is None:
                    cw["train_btn"] = self._make_button(
                        cw["btn_row"], "🧠 Train", COLORS["training"],
                        lambda i=idx: self._on_start_training(i),
                        font=self.font_small
                    )
                    cw["train_btn"].pack(side="left", padx=(0, 10))

            # Show/hide log button
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            log_exists = (self._get_data_dir() / safe_name / "train.log").exists()
            if log_exists or training_status in ("training", "trained", "failed"):
                if cw["log_btn"] is None:
                    cw["log_btn"] = self._make_button(
                        cw["btn_row"], "📄 Log", COLORS["accent"],
                        lambda i=idx: self._show_training_log(i),
                        font=self.font_small
                    )
                    cw["log_btn"].pack(side="left", padx=(0, 10))
            else:
                if cw["log_btn"] is not None:
                    cw["log_btn"].destroy()
                    cw["log_btn"] = None

        self._overview_refresh_id = self.root.after(2000, self._schedule_overview_refresh)

    def _select_task(self, task_idx, mode="collect"):
        self.current_task = task_idx
        if self._overview_refresh_id:
            self.root.after_cancel(self._overview_refresh_id)
            self._overview_refresh_id = None

        if mode == "policy":
            self._start_policy_execution()
        else:
            self._show_idle()

    def _get_data_dir(self):
        """Return the base data directory (matches DATA_DIR in record script)."""
        from pathlib import Path
        return Path("~/datasets").expanduser()

    # ================================================================
    # Training Controls
    # ================================================================

    def _on_start_training(self, task_idx):
        """User clicked Train button for a task."""
        task = self.task_configs[task_idx]
        name = task["name"]
        self.events["start_training_task"] = name

    def _show_training_log(self, task_idx):
        """Show the training log viewer for a task."""
        self.state = "training_log"
        if self._overview_refresh_id:
            self.root.after_cancel(self._overview_refresh_id)
            self._overview_refresh_id = None
        self._clear_content()

        task = self.task_configs[task_idx]
        name = task["name"]
        self._log_task_name = name
        training_status = self.task_state.get(name, {}).get("training_status", "not_started")

        badge_text, badge_color = self._training_badge(training_status)
        self.subtitle_label.config(text=f"Training Log — {name}  {badge_text}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=20, pady=15)
        card.pack(expand=True, fill="both")

        # Log text area with scrollbar
        log_frame = tk.Frame(card, bg=COLORS["card"])
        log_frame.pack(expand=True, fill="both", pady=(5, 10))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self._log_text = tk.Text(
            log_frame, bg="#0d1117", fg="#c9d1d9",
            font=("Courier", 11), wrap="word",
            yscrollcommand=scrollbar.set, state="disabled",
            borderwidth=0, highlightthickness=0
        )
        self._log_text.pack(expand=True, fill="both")
        scrollbar.config(command=self._log_text.yview)

        # Bottom buttons
        btn_frame = tk.Frame(card, bg=COLORS["card"])
        btn_frame.pack(fill="x", pady=(5, 0))

        self._make_button(
            btn_frame, "← Back", COLORS["button_back"], self._close_training_log,
            font=self.font_small
        ).pack(side="left", padx=(0, 10))

        self._load_log_content(name)
        self._schedule_log_refresh(name)

    def _load_log_content(self, task_name):
        """Read the training log file and display it."""
        safe_name = task_name.lower().replace(" ", "_").replace("/", "_")
        log_path = self._get_data_dir() / safe_name / "train.log"

        if self._log_text is None:
            return

        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")

        if log_path.exists():
            try:
                content = log_path.read_text(errors="replace")
                self._log_text.insert("end", content)
            except Exception as e:
                self._log_text.insert("end", f"Error reading log: {e}")
        else:
            self._log_text.insert("end", "No training log file yet.\n\nThe log will appear here once training starts.")

        self._log_text.config(state="disabled")
        self._log_text.see("end")  # Auto-scroll to bottom

    def _schedule_log_refresh(self, task_name):
        """Refresh log content periodically while viewing."""
        if self.state != "training_log" or self._log_task_name != task_name:
            return

        # Also update the subtitle badge
        training_status = self.task_state.get(task_name, {}).get("training_status", "not_started")
        badge_text, badge_color = self._training_badge(training_status)
        self.subtitle_label.config(text=f"Training Log — {task_name}  {badge_text}")

        self._load_log_content(task_name)
        self._log_refresh_id = self.root.after(3000, lambda: self._schedule_log_refresh(task_name))

    def _close_training_log(self):
        """Return from log viewer to task overview."""
        if self._log_refresh_id:
            self.root.after_cancel(self._log_refresh_id)
            self._log_refresh_id = None
        self._log_text = None
        self._log_task_name = None
        self._show_task_overview()

    # ================================================================
    # Data Collection Screens (idle → recording → review → saving)
    # ================================================================

    def _show_idle(self):
        self.state = "idle"
        self._clear_content()
        task = self.task_configs[self.current_task]
        state = self.task_state.get(task["name"], {})
        collected = state.get("episodes_collected", 0)
        required = task["required_episodes"]

        self.subtitle_label.config(
            text=f"Task: {task['name']}  —  {collected}/{required} episodes"
        )

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text="Your task:", font=self.font_medium,
            fg=COLORS["text_dim"], bg=COLORS["card"]
        ).pack(pady=(10, 0))
        tk.Label(
            card, text=task["name"], font=self.font_large,
            fg=COLORS["warning"], bg=COLORS["card"], wraplength=700
        ).pack(pady=(5, 5))
        if task.get("description"):
            tk.Label(
                card, text=task["description"], font=self.font_small,
                fg=COLORS["text_dim"], bg=COLORS["card"], wraplength=700
            ).pack(pady=(0, 20))

        tk.Label(
            card, text="Press the button below when you are ready to start!",
            font=self.font_large, fg=COLORS["text"], bg=COLORS["card"], wraplength=700
        ).pack(pady=(10, 10))

        tk.Label(
            card, text="Welcome!\nHelp us teach the robot by demonstrating a task.",
            font=self.font_small, fg=COLORS["text_dim"], bg=COLORS["card"], wraplength=700
        ).pack(pady=(0, 20))

        btn_frame = tk.Frame(card, bg=COLORS["card"])
        btn_frame.pack(pady=(10, 20))

        self._make_button(
            btn_frame, "▶  Start Recording", COLORS["button_start"], self._on_start
        ).pack(side="left", padx=15)

        self._make_button(
            btn_frame, "← Back", COLORS["button_back"], self._go_back,
            font=self.font_small
        ).pack(side="left", padx=15)

    def _show_recording(self):
        self.state = "recording"
        self._clear_content()
        task = self.task_configs[self.current_task]

        self.subtitle_label.config(text=f"Recording — {task['name']}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text="⏺  Recording... Perform the task now!",
            font=self.font_large, fg=COLORS["text"], bg=COLORS["card"], wraplength=700
        ).pack(pady=(20, 10))

        self.recording_tip_label = tk.Label(
            card, text="Use the leader arm to guide the robot.\nPress 'Stop Recording' when you are done.",
            font=self.font_small, fg=COLORS["text_dim"], bg=COLORS["card"], wraplength=700
        )
        self.recording_tip_label.pack(pady=(0, 20))

        btn_frame = tk.Frame(card, bg=COLORS["card"])
        btn_frame.pack(pady=(10, 20))

        self._make_button(
            btn_frame, "⏹  Stop Recording", COLORS["button_stop"], self._on_stop
        ).pack()

        self._remaining_s = task.get("max_episode_time_s", 60)
        self._update_timer()

    def _update_timer(self):
        if self.state != "recording":
            return
        if self._remaining_s <= 0:
            self._on_timeout()
            return
        minutes = self._remaining_s // 60
        seconds = self._remaining_s % 60
        self.recording_tip_label.config(
            text="Use the leader arm to guide the robot.\nPress 'Stop Recording' when you are done.\n"
                 f"Time remaining: {minutes}:{seconds:02d}"
        )
        self._remaining_s -= 1
        self._timer_id = self.root.after(1000, self._update_timer)

    def _cancel_timer(self):
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

    def _show_timeout(self):
        self.state = "timeout"
        self._clear_content()
        task = self.task_configs[self.current_task]
        self.subtitle_label.config(text=f"Timeout — {task['name']}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text="⏰  Time's up!\nThe trial took too long and was discarded.",
            font=self.font_large, fg=COLORS["warning"], bg=COLORS["card"], wraplength=700
        ).pack(expand=True)

        self.root.after(4000, self._show_idle)

    def _show_review(self):
        self.state = "review"
        self._clear_content()
        task = self.task_configs[self.current_task]
        self.subtitle_label.config(text=f"Review — {task['name']}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text="Was the demonstration successful?",
            font=self.font_large, fg=COLORS["text"], bg=COLORS["card"], wraplength=700
        ).pack(pady=(20, 30))

        btn_frame = tk.Frame(card, bg=COLORS["card"])
        btn_frame.pack(pady=(10, 20))

        self._make_button(
            btn_frame, "✓  Yes, save it!", COLORS["button_yes"], self._on_accept
        ).pack(side="left", padx=15)
        self._make_button(
            btn_frame, "✗  No, let me retry", COLORS["button_no"], self._on_reject
        ).pack(side="left", padx=15)

    def _show_saving(self, message):
        self.state = "saving"
        self._clear_content()
        task = self.task_configs[self.current_task]
        self.subtitle_label.config(text=f"{task['name']}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text=message, font=self.font_large,
            fg=COLORS["text"], bg=COLORS["card"]
        ).pack(expand=True)

    # ================================================================
    # Policy Execution Screen
    # ================================================================

    def _start_policy_execution(self):
        task = self.task_configs[self.current_task]
        self.events["run_policy_task"] = task["name"]
        self._show_policy_running()

    def _show_policy_running(self):
        self.state = "policy_running"
        self._clear_content()
        task = self.task_configs[self.current_task]
        self.subtitle_label.config(text=f"Policy Execution — {task['name']}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text="🤖  Policy is running...",
            font=self.font_large, fg=COLORS["button_policy"], bg=COLORS["card"]
        ).pack(pady=(20, 10))
        tk.Label(
            card, text=f"The robot is autonomously performing:\n{task['name']}",
            font=self.font_medium, fg=COLORS["text"], bg=COLORS["card"], wraplength=700
        ).pack(pady=(10, 20))

        btn_frame = tk.Frame(card, bg=COLORS["card"])
        btn_frame.pack(pady=(10, 20))

        self._make_button(
            btn_frame, "⏹  Stop Policy", COLORS["button_stop"], self._on_stop_policy
        ).pack()

    # ================================================================
    # Button Callbacks
    # ================================================================

    def _on_start(self):
        if self.state != "idle":
            return
        task = self.task_configs[self.current_task]
        self.events["start_episode"] = True
        self.events["current_task_name"] = task["name"]
        self._show_recording()

    def _on_stop(self):
        if self.state != "recording":
            return
        self._cancel_timer()
        self.events["exit_early"] = True
        self._show_review()

    def _on_timeout(self):
        if self.state != "recording":
            return
        self._cancel_timer()
        self.events["exit_early"] = True
        self.events["rerecord_episode"] = True
        self.events["episode_accepted"] = False
        self._show_timeout()

    def _on_accept(self):
        if self.state != "review":
            return
        self._show_saving("Saving episode...")
        self.events["episode_accepted"] = True

        task = self.task_configs[self.current_task]
        name = task["name"]
        ts = self.task_state.setdefault(name, {"episodes_collected": 0, "training_status": "not_started", "policy_path": None})
        ts["episodes_collected"] = ts.get("episodes_collected", 0) + 1

        self.root.after(1500, self._after_save)

    def _after_save(self):
        task = self.task_configs[self.current_task]
        state = self.task_state.get(task["name"], {})
        if state.get("episodes_collected", 0) >= task["required_episodes"]:
            # All episodes for this task collected
            self._show_saving("All episodes collected for this task!")
            self.root.after(2000, self._show_task_overview)
        else:
            self._show_idle()

    def _on_reject(self):
        if self.state != "review":
            return
        self._show_failure_reason()

    def _show_failure_reason(self):
        """Show multiple-choice screen for why the trial failed."""
        self.state = "failure_reason"
        self._clear_content()
        task = self.task_configs[self.current_task]
        self.subtitle_label.config(text=f"Feedback — {task['name']}")

        card = tk.Frame(self.content_frame, bg=COLORS["card"], padx=40, pady=30)
        card.pack(expand=True, fill="both")

        tk.Label(
            card, text="What went wrong?",
            font=self.font_large, fg=COLORS["text"], bg=COLORS["card"]
        ).pack(pady=(20, 20))

        reasons = [
            "Robot did not grab the object",
            "Robot dropped the object",
            "Robot moved to the wrong position",
            "Robot collided with something",
            "I made a mistake operating the leader arm",
            "The task took too long",
            "Other / not sure",
        ]

        for reason in reasons:
            self._make_button(
                card, reason, COLORS["accent"],
                lambda r=reason: self._on_reason_selected(r),
                font=self.font_small
            ).pack(fill="x", pady=4, padx=40)

    def _on_reason_selected(self, reason):
        """User picked a failure reason."""
        if self.state != "failure_reason":
            return
        self._show_saving(f"Saving failed trial...\nReason: {reason}")
        self.events["reject_reason"] = reason
        self.events["rerecord_episode"] = True
        self.events["exit_early"] = True
        self.events["episode_accepted"] = False
        self.root.after(1500, self._show_idle)

    def _on_stop_policy(self):
        if self.state != "policy_running":
            return
        self.events["exit_early"] = True
        self.events["stop_policy"] = True
        self._show_saving("Stopping policy...")
        self.root.after(2000, self._show_task_overview)

    def _go_back(self):
        self._cancel_timer()
        self._show_task_overview()

    def _on_quit(self):
        self._cancel_timer()
        if self._overview_refresh_id:
            self.root.after_cancel(self._overview_refresh_id)
        if self._log_refresh_id:
            self.root.after_cancel(self._log_refresh_id)
        self.events["stop_recording"] = True
        self.events["exit_early"] = True
        self.root.quit()
        self.root.destroy()

    # ================================================================
    # Public API
    # ================================================================

    def run(self):
        """Start the GUI main loop (blocking). Call from main thread."""
        self.root.mainloop()

    def notify_episode_saved(self):
        """Called from recording thread after episode is persisted."""

    def notify_episode_discarded(self):
        """Called from recording thread after episode is discarded."""

    def notify_policy_finished(self):
        """Called when policy execution finishes."""
        self.root.after(0, self._show_task_overview)


def init_faire_multi_task_gui(task_configs: list[dict], task_state: dict):
    """
    Initialize the multi-task faire GUI.

    Returns:
        (gui, events) tuple. gui.run() must be called from main thread.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "start_episode": False,
        "episode_accepted": False,
        "current_task_name": None,
        "run_policy_task": None,
        "stop_policy": False,
        "start_training_task": None,
        "reject_reason": None,
    }
    gui = FaireMultiTaskGUI(events, task_configs, task_state)
    return gui, events
