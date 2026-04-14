"""
Faire Booth GUI for SO-101 Robot Data Collection

A tkinter-based GUI that guides users through the data collection process
at a faire/exhibition booth. Designed for autonomous operation with minimal
supervision.

The GUI controls recording events and replaces the keyboard listener.
"""

import tkinter as tk
from tkinter import font as tkfont


# Color scheme
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
}

# UI text - easy to translate / customize
STRINGS = {
    "title": "🤖 Robot Data Collection",
    "welcome": "Welcome!\nHelp us teach the robot by demonstrating a task.",
    "task_label": "Your task:",
    "episode_progress": "Episode {current} of {total}",
    "state_idle": "Press the button below when you are ready to start!",
    "state_recording": "⏺  Recording... Perform the task now!",
    "state_review": "Was the demonstration successful?",
    "state_done": "Thank you for contributing!\nAll episodes have been recorded.",
    "state_timeout": "⏰  Time's up!\nThe trial took too long and was discarded.",
    "btn_start": "▶  Start Recording",
    "btn_stop": "⏹  Stop Recording",
    "btn_yes": "✓  Yes, save it!",
    "btn_no": "✗  No, let me retry",
    "btn_quit": "Exit",
    "recording_tip": "Use the leader arm to guide the robot.\nPress 'Stop Recording' when you are done.",
    "timer_format": "Time remaining: {minutes}:{seconds:02d}",
    "saving": "Saving episode...",
    "discarding": "Discarding episode, you can try again.",
}


class FaireGUI:
    """
    A full-screen GUI for guiding users through robot data collection.

    States:
        idle      - Waiting for user to start an episode
        recording - Episode is being recorded
        review    - User decides if episode was successful
        saving    - Brief state while episode is being saved
        done      - All episodes recorded
    """

    def __init__(self, events: dict, task_description: str, num_episodes: int, max_episode_time_s: int = 60):
        """
        Args:
            events: Shared event dict (same format as keyboard listener events).
            task_description: The task the user should perform.
            num_episodes: Total number of episodes to collect.
            max_episode_time_s: Maximum time per episode in seconds.
        """
        self.events = events
        self.task_description = task_description
        self.num_episodes = num_episodes
        self.max_episode_time_s = max_episode_time_s
        self.current_episode = 0
        self.state = "idle"
        self._timer_id = None

        self._build_ui()

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Robot Data Collection Booth")
        self.root.configure(bg=COLORS["bg"])
        # Start maximized but allow resize
        self.root.attributes("-zoomed", True)
        self.root.minsize(800, 600)

        # Fonts
        self.font_title = tkfont.Font(family="Helvetica", size=36, weight="bold")
        self.font_large = tkfont.Font(family="Helvetica", size=24)
        self.font_medium = tkfont.Font(family="Helvetica", size=18)
        self.font_button = tkfont.Font(family="Helvetica", size=22, weight="bold")
        self.font_small = tkfont.Font(family="Helvetica", size=14)

        # Main container
        container = tk.Frame(self.root, bg=COLORS["bg"])
        container.pack(expand=True, fill="both", padx=40, pady=30)

        # Title
        tk.Label(
            container, text=STRINGS["title"],
            font=self.font_title, fg=COLORS["text"], bg=COLORS["bg"]
        ).pack(pady=(0, 10))

        # Progress label
        self.progress_label = tk.Label(
            container, text="", font=self.font_medium,
            fg=COLORS["text_dim"], bg=COLORS["bg"]
        )
        self.progress_label.pack(pady=(0, 20))

        # Card frame for main content
        self.card = tk.Frame(container, bg=COLORS["card"], padx=40, pady=30)
        self.card.pack(expand=True, fill="both")

        # Task description
        tk.Label(
            self.card, text=STRINGS["task_label"],
            font=self.font_medium, fg=COLORS["text_dim"], bg=COLORS["card"]
        ).pack(pady=(10, 0))
        tk.Label(
            self.card, text=self.task_description,
            font=self.font_large, fg=COLORS["warning"], bg=COLORS["card"],
            wraplength=700
        ).pack(pady=(5, 20))

        # Status message
        self.status_label = tk.Label(
            self.card, text="", font=self.font_large,
            fg=COLORS["text"], bg=COLORS["card"], wraplength=700
        )
        self.status_label.pack(pady=(10, 10))

        # Tip / secondary info
        self.tip_label = tk.Label(
            self.card, text="", font=self.font_small,
            fg=COLORS["text_dim"], bg=COLORS["card"], wraplength=700
        )
        self.tip_label.pack(pady=(0, 20))

        # Button frame
        self.btn_frame = tk.Frame(self.card, bg=COLORS["card"])
        self.btn_frame.pack(pady=(10, 20))

        # Quit button (always visible, bottom-right)
        quit_frame = tk.Frame(container, bg=COLORS["bg"])
        quit_frame.pack(fill="x", pady=(10, 0))
        self.quit_btn = tk.Button(
            quit_frame, text=STRINGS["btn_quit"],
            font=self.font_small, bg=COLORS["button_quit"], fg=COLORS["text"],
            activebackground="#a93226", activeforeground=COLORS["text"],
            relief="flat", padx=20, pady=8,
            command=self._on_quit
        )
        self.quit_btn.pack(side="right")

        self._update_progress()
        self._show_idle()

        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _clear_buttons(self):
        for widget in self.btn_frame.winfo_children():
            widget.destroy()

    def _make_button(self, text, color, command):
        btn = tk.Button(
            self.btn_frame, text=text,
            font=self.font_button, bg=color, fg=COLORS["text"],
            activebackground=color, activeforeground=COLORS["text"],
            relief="flat", padx=40, pady=20,
            command=command
        )
        btn.pack(side="left", padx=15)
        return btn

    def _update_progress(self):
        self.progress_label.config(
            text=STRINGS["episode_progress"].format(
                current=self.current_episode + 1, total=self.num_episodes
            )
        )

    # ---- State transitions ----

    def _show_idle(self):
        self.state = "idle"
        self._clear_buttons()
        self.status_label.config(text=STRINGS["state_idle"])
        self.tip_label.config(text=STRINGS["welcome"])
        self._make_button(STRINGS["btn_start"], COLORS["button_start"], self._on_start)

    def _show_recording(self):
        self.state = "recording"
        self._clear_buttons()
        self.status_label.config(text=STRINGS["state_recording"])
        self.tip_label.config(text=STRINGS["recording_tip"])
        self._make_button(STRINGS["btn_stop"], COLORS["button_stop"], self._on_stop)
        self._remaining_s = self.max_episode_time_s
        self._update_timer()

    def _update_timer(self):
        """Update the countdown timer during recording."""
        if self.state != "recording":
            return
        if self._remaining_s <= 0:
            self._on_timeout()
            return
        minutes = self._remaining_s // 60
        seconds = self._remaining_s % 60
        self.tip_label.config(
            text=STRINGS["recording_tip"] + "\n" +
            STRINGS["timer_format"].format(minutes=minutes, seconds=seconds)
        )
        self._remaining_s -= 1
        self._timer_id = self.root.after(1000, self._update_timer)

    def _cancel_timer(self):
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

    def _show_timeout(self):
        self.state = "timeout"
        self._clear_buttons()
        self.status_label.config(text=STRINGS["state_timeout"])
        self.tip_label.config(text="")

    def _show_review(self):
        self.state = "review"
        self._clear_buttons()
        self.status_label.config(text=STRINGS["state_review"])
        self.tip_label.config(text="")
        self._make_button(STRINGS["btn_yes"], COLORS["button_yes"], self._on_accept)
        self._make_button(STRINGS["btn_no"], COLORS["button_no"], self._on_reject)

    def _show_saving(self, message):
        self.state = "saving"
        self._clear_buttons()
        self.status_label.config(text=message)
        self.tip_label.config(text="")

    def _show_done(self):
        self.state = "done"
        self._clear_buttons()
        self.status_label.config(text=STRINGS["state_done"])
        self.tip_label.config(text="")

    # ---- Button callbacks ----

    def _on_start(self):
        if self.state != "idle":
            return
        self.events["start_episode"] = True
        self._show_recording()

    def _on_stop(self):
        if self.state != "recording":
            return
        self._cancel_timer()
        self.events["exit_early"] = True
        self._show_review()

    def _on_timeout(self):
        """Called when the episode timer runs out."""
        if self.state != "recording":
            return
        self._cancel_timer()
        self.events["exit_early"] = True
        self.events["rerecord_episode"] = True
        self.events["episode_accepted"] = False
        self._show_timeout()
        self.root.after(4000, self._show_idle)

    def _on_accept(self):
        if self.state != "review":
            return
        self._show_saving(STRINGS["saving"])
        # Signal that the episode should be saved (not rerecord)
        self.events["episode_accepted"] = True
        self.current_episode += 1
        if self.current_episode >= self.num_episodes:
            self.root.after(1000, self._show_done)
        else:
            self._update_progress()
            self.root.after(1500, self._show_idle)

    def _on_reject(self):
        if self.state != "review":
            return
        self._show_saving(STRINGS["discarding"])
        # Signal rerecord
        self.events["rerecord_episode"] = True
        self.events["exit_early"] = True
        self.events["episode_accepted"] = False
        self.root.after(1500, self._show_idle)

    def _on_quit(self):
        self.events["stop_recording"] = True
        self.events["exit_early"] = True
        self.root.quit()
        self.root.destroy()

    # ---- Public API ----

    def run(self):
        """Start the GUI main loop (blocking). Call from main thread."""
        self.root.mainloop()

    def notify_episode_saved(self):
        """Called from recording thread after episode is saved."""

    def notify_episode_discarded(self):
        """Called from recording thread after episode is discarded."""


def init_faire_gui(task_description: str, num_episodes: int, max_episode_time_s: int = 60):
    """
    Initialize the faire GUI and return events dict compatible with
    the keyboard listener interface.

    Returns:
        (gui, events) tuple. gui.run() must be called from main thread.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "start_episode": False,
        "episode_accepted": False,
    }
    gui = FaireGUI(events, task_description, num_episodes, max_episode_time_s)
    return gui, events
