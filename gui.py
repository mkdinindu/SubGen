#!/usr/bin/env python3
"""GUI frontend for SubGen — subtitle generator with Whisper + Ollama."""

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from translate_srt import default_output_path

try:
    import sv_ttk
except ImportError:  # GUI still works with the stock ttk theme
    sv_ttk = None


# Display name -> Whisper language code. The GUI shows full names while the CLI
# still receives the short codes Whisper expects.
LANGUAGE_CODES = {
    "Auto-detect": "",
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Japanese": "ja", "Korean": "ko", "Chinese": "zh", "Arabic": "ar",
    "Hindi": "hi", "Turkish": "tr", "Polish": "pl", "Swedish": "sv",
    "Danish": "da", "Finnish": "fi", "Norwegian": "no", "Czech": "cs",
    "Hungarian": "hu", "Romanian": "ro", "Greek": "el", "Hebrew": "he",
    "Thai": "th", "Vietnamese": "vi", "Indonesian": "id", "Malay": "ms",
}
LANGUAGES = list(LANGUAGE_CODES)


def language_code(value):
    """Return the Whisper code for a language display name (or a raw code)."""
    value = (value or "").strip()
    if not value or value == "Auto-detect":
        return ""
    return LANGUAGE_CODES.get(value, value)


def resource_path():
    return os.path.dirname(os.path.abspath(__file__))


class SubGenGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SubGen — Subtitle Generator")
        self.root.geometry("760x720")
        self.root.minsize(620, 560)
        self.root.resizable(True, True)
        self._configure_styles()
        self.proc = None
        self.trans_frame = None
        self.log_queue = queue.Queue()
        self.build_widgets()

    def _configure_styles(self):
        """Use sv-ttk for rounded, native-looking fields/buttons when available."""
        if sv_ttk is None:
            self.accent = "#4aa3df"
            self.bg_color = "#1e1e1e"
            self.fg_color = "#d4d4d4"
            self.sel_color = "#2a5d7a"
        else:
            sv_ttk.set_theme("dark", root=self.root)
            self.accent = self.root.tk.call("set", "ttk::theme::sv_dark::colors(-accent)")
            self.bg_color = self.root.tk.call("set", "ttk::theme::sv_dark::colors(-bg)")
            self.fg_color = self.root.tk.call("set", "ttk::theme::sv_dark::colors(-fg)")
            self.sel_color = self.root.tk.call("set", "ttk::theme::sv_dark::colors(-selbg)")

        style = ttk.Style()
        style.configure("TEntry", font=("Segoe UI", 10))
        style.configure("TCombobox", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"),
                        foreground=self.accent)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 9))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", font=("Segoe UI", 10, "bold"),
                        foreground="#5ad19b")
        style.configure("Error.TLabel", font=("Segoe UI", 10, "bold"),
                        foreground="#ff7b7b")

    def build_widgets(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)

        # ── Header ──
        header = ttk.Frame(main, padding=(12, 10))
        header.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="SubGen", style="Title.TLabel").grid(
            row=0, column=0, sticky=tk.W)
        ttk.Label(header, text="Whisper transcription  •  Ollama translation",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        ttk.Separator(header, orient=tk.HORIZONTAL).grid(
            row=2, column=0, sticky=tk.EW, pady=(8, 0))

        # ── Mode ──
        self.mode_var = tk.StringVar(value="video")
        mode_frame = ttk.Frame(main)
        mode_frame.grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Video → SRT (Whisper)",
                        variable=self.mode_var, value="video",
                        command=self.toggle_mode).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(mode_frame, text="Existing SRT → translated SRT",
                        variable=self.mode_var, value="srt",
                        command=self.toggle_mode).pack(side=tk.LEFT, padx=(8, 0))

        # ── Input file ──
        self.input_label = ttk.Label(main, text="Media File:")
        self.input_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 2))
        row1 = ttk.Frame(main)
        row1.grid(row=3, column=0, sticky=tk.EW, pady=(0, 8))
        row1.columnconfigure(0, weight=1)
        self.input_path = tk.StringVar()
        ttk.Entry(row1, textvariable=self.input_path).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(row1, text="Browse", command=self.browse_input).grid(row=0, column=1, padx=(6, 0))

        # ── Whisper settings ──
        self.model_var = tk.StringVar(value="base")
        self.lang_var = tk.StringVar(value="Auto-detect")
        self.output_var = tk.StringVar(value="")

        opts_frame = ttk.LabelFrame(main, text="Whisper Settings", padding=8)
        self.whisper_frame = opts_frame
        opts_frame.grid(row=4, column=0, sticky=tk.EW, pady=(0, 8))
        opts_frame.columnconfigure(1, weight=1)

        ttk.Label(opts_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(opts_frame, textvariable=self.model_var,
                      values=["tiny", "base", "small", "medium", "large"],
                      state="readonly", width=14).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(opts_frame, text="Audio Language:").grid(row=0, column=2, sticky=tk.W, padx=(12, 6))
        lang_cb = ttk.Combobox(opts_frame, textvariable=self.lang_var,
                               values=LANGUAGES, state="readonly", width=18)
        lang_cb.grid(row=0, column=3, sticky=tk.W)

        ttk.Label(opts_frame, text="Output:").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        ttk.Entry(opts_frame, textvariable=self.output_var).grid(
            row=1, column=1, columnspan=3, sticky=tk.EW, pady=(6, 0))

        # ── Timing ──
        timing_frame = ttk.LabelFrame(main, text="Timing", padding=8)
        self.timing_frame = timing_frame
        timing_frame.grid(row=5, column=0, sticky=tk.EW, pady=(0, 8))

        self.delay_var = tk.StringVar(value="0.0")
        self.min_dur_var = tk.StringVar(value="1.0")
        self.max_dur_var = tk.StringVar(value="6.0")
        self.min_gap_var = tk.StringVar(value="0.0")
        self.chars_var = tk.StringVar(value="42")
        self.max_lines_var = tk.StringVar(value="2")

        ttk.Label(timing_frame, text="Delay (s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(timing_frame, textvariable=self.delay_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(timing_frame, text="Min Dur (s):").grid(row=0, column=2, sticky=tk.W, padx=(12, 4))
        ttk.Entry(timing_frame, textvariable=self.min_dur_var, width=8).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(timing_frame, text="Max Dur (s):").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(timing_frame, textvariable=self.max_dur_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Label(timing_frame, text="Min Gap (s):").grid(row=1, column=2, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(timing_frame, textvariable=self.min_gap_var, width=8).grid(row=1, column=3, sticky=tk.W, pady=(6, 0))
        ttk.Label(timing_frame, text="Chars/Line:").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(timing_frame, textvariable=self.chars_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Label(timing_frame, text="Max Lines:").grid(row=2, column=2, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(timing_frame, textvariable=self.max_lines_var, width=8).grid(row=2, column=3, sticky=tk.W, pady=(6, 0))

        # ── Translation ──
        self.translate_var = tk.BooleanVar(value=False)
        self.trans_frame = ttk.LabelFrame(main, text="Translation (Ollama)", padding=8)
        self.trans_frame.grid(row=6, column=0, sticky=tk.EW, pady=(0, 8))
        self.trans_frame.columnconfigure(1, weight=1)
        self.trans_frame.columnconfigure(4, weight=1)

        self.translate_check = ttk.Checkbutton(
            self.trans_frame, text="Enable Translation",
            variable=self.translate_var, command=self.toggle_translate
        )
        self.translate_check.grid(row=0, column=0, columnspan=6, sticky=tk.W)

        self.src_lang_var = tk.StringVar(value="English")
        self.src_code_var = tk.StringVar(value="en")
        self.tgt_lang_var = tk.StringVar(value="Spanish")
        self.tgt_code_var = tk.StringVar(value="es")
        self.ollama_var = tk.StringVar(value="gemma")
        self.ollama_url_var = tk.StringVar(value="")
        self.batch_size_var = tk.StringVar(value="20")
        self.timeout_var = tk.StringVar(value="300")
        self.retries_var = tk.StringVar(value="3")

        ttk.Label(self.trans_frame, text="Source:").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.src_lang_var, width=14).grid(
            row=1, column=1, sticky=tk.W, pady=(6, 0))
        self.src_code_label = ttk.Label(self.trans_frame, text=f"({self.src_code_var.get()})")
        self.src_code_label.grid(row=1, column=2, sticky=tk.W, padx=(2, 0), pady=(6, 0))
        self.src_code_var.trace_add("write",
                                     lambda *a: self.src_code_label.configure(text=f"({self.src_code_var.get()})"))

        ttk.Label(self.trans_frame, text="Target:").grid(row=2, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.tgt_lang_var, width=14).grid(
            row=2, column=1, sticky=tk.W, pady=(6, 0))
        self.tgt_code_label = ttk.Label(self.trans_frame, text=f"({self.tgt_code_var.get()})")
        self.tgt_code_label.grid(row=2, column=2, sticky=tk.W, padx=(2, 0), pady=(6, 0))
        self.tgt_code_var.trace_add("write",
                                     lambda *a: self.tgt_code_label.configure(text=f"({self.tgt_code_var.get()})"))

        ttk.Label(self.trans_frame, text="Ollama Model:").grid(row=1, column=3, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.ollama_var, width=12).grid(
            row=1, column=4, sticky=tk.W, pady=(6, 0))

        ttk.Label(self.trans_frame, text="Timeout (s):").grid(row=2, column=3, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.timeout_var, width=8).grid(
            row=2, column=4, sticky=tk.W, pady=(6, 0))

        ttk.Label(self.trans_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.batch_size_var, width=8).grid(
            row=3, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Label(self.trans_frame, text="Retries:").grid(row=3, column=2, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.retries_var, width=8).grid(
            row=3, column=3, sticky=tk.W, pady=(6, 0))

        ttk.Label(self.trans_frame, text="Server URL:").grid(row=4, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(self.trans_frame, textvariable=self.ollama_url_var).grid(
            row=4, column=1, columnspan=5, sticky=tk.EW, pady=(6, 0))

        self.toggle_translate()

        # ── Progress & Log ──
        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.grid(row=7, column=0, sticky=tk.EW, pady=(0, 6))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=8, column=0, sticky=tk.W, pady=(0, 4))
        main.rowconfigure(9, weight=1)

        log_frame = ttk.LabelFrame(main, text="Output Log", padding=4)
        log_frame.grid(row=9, column=0, sticky=tk.NSEW, pady=(0, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED,
                                bg=self.bg_color, fg=self.fg_color,
                                font=("Consolas", 9), relief=tk.FLAT,
                                borderwidth=0, highlightthickness=0,
                                insertbackground=self.fg_color,
                                selectbackground=self.sel_color,
                                selectforeground="#ffffff")
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        # ── Buttons ──
        btn_row = ttk.Frame(main)
        btn_row.grid(row=10, column=0, sticky=tk.EW)
        ttk.Button(btn_row, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        self.run_btn = ttk.Button(btn_row, text="Generate Subtitles", command=self.run,
                                  style="Accent.TButton", width=24)
        self.run_btn.pack(side=tk.RIGHT)

        self.root.after(100, self._poll_log)

        self.toggle_mode()

    # ── helpers ──

    def _poll_log(self):
        """Marshal worker-thread messages onto the Tk main thread."""
        try:
            while True:
                kind, value = self.log_queue.get_nowait()
                if kind == "log":
                    self.log(value)
                elif kind == "status":
                    self.set_status(value)
                elif kind == "finish":
                    self.finish()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def is_srt_mode(self):
        return self.mode_var.get() == "srt"

    def browse_input(self):
        if self.is_srt_mode():
            path = filedialog.askopenfilename(
                title="Select SRT File",
                filetypes=[("SubRip subtitles", "*.srt"), ("All files", "*.*")]
            )
        else:
            path = filedialog.askopenfilename(
                title="Select Audio or Video File",
                filetypes=[
                    ("Media files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                    ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
                    ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                    ("All files", "*.*"),
                ]
            )
        if path:
            self.input_path.set(path)
            if self.is_srt_mode():
                self.output_var.set(default_output_path(path, self.tgt_code_var.get()))
            else:
                self.output_var.set(os.path.splitext(path)[0] + ".srt")

    def toggle_mode(self):
        srt_mode = self.is_srt_mode()
        self.input_label.configure(text="SRT File:" if srt_mode else "Media File:")
        self.run_btn.configure(text="Translate SRT" if srt_mode else "Generate Subtitles")

        if srt_mode:
            self.whisper_frame.grid_remove()
            self.timing_frame.grid_remove()
            self.translate_check.grid_remove()
            if self.input_path.get().lower().endswith(".srt"):
                self.output_var.set(
                    default_output_path(self.input_path.get(), self.tgt_code_var.get())
                )
        else:
            self.whisper_frame.grid(row=4, column=0, sticky=tk.EW, pady=(0, 8))
            self.timing_frame.grid(row=5, column=0, sticky=tk.EW, pady=(0, 8))
            self.translate_check.grid(row=0, column=0, columnspan=6, sticky=tk.W)
        self.toggle_translate()

    def toggle_translate(self):
        # SRT mode always translates, so the checkbox is hidden there.
        enabled = self.translate_var.get() or self.is_srt_mode()
        state = tk.NORMAL if enabled else tk.DISABLED
        if not self.trans_frame:
            return
        for child in self.trans_frame.winfo_children():
            if isinstance(child, ttk.Checkbutton):
                continue
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()

    def set_status(self, msg):
        self.status_var.set(msg)
        if "Done" in msg:
            style = "Success.TLabel"
        elif "Failed" in msg or "Error" in msg:
            style = "Error.TLabel"
        else:
            style = "Status.TLabel"
        self.status_label.configure(style=style)
        self.root.update_idletasks()

    # ── run pipeline ──

    def build_args(self, input_path):
        if self.is_srt_mode():
            return self.build_srt_args(input_path)
        return self.build_video_args(input_path)

    def build_translation_args(self):
        args = ["--source-lang", self.src_lang_var.get().strip()]
        args += ["--source-code", self.src_code_var.get().strip()]
        args += ["--target-lang", self.tgt_lang_var.get().strip()]
        args += ["--target-code", self.tgt_code_var.get().strip()]
        args += ["--ollama-model", self.ollama_var.get().strip()]
        url = self.ollama_url_var.get().strip()
        if url:
            args += ["--ollama-url", url]
        args += ["--batch-size", self.batch_size_var.get().strip() or "20"]
        args += ["--ollama-timeout", self.timeout_var.get().strip() or "300"]
        args += ["--ollama-retries", self.retries_var.get().strip() or "3"]
        return args

    def build_srt_args(self, input_path):
        args = [input_path]
        out = self.output_var.get().strip()
        if out:
            args += ["-o", out]
        return args + self.build_translation_args()

    def build_video_args(self, video):
        args = [video]
        model = self.model_var.get()
        if model != "base":
            args += ["--whisper-model", model]
        lang = language_code(self.lang_var.get())
        if lang:
            args += ["--language", lang]
        out = self.output_var.get().strip()
        if out:
            args += ["-o", out]
        delay = self.delay_var.get().strip()
        if delay and delay != "0.0":
            args += ["--delay", delay]
        args += ["--min-duration", self.min_dur_var.get().strip() or "1.0"]
        args += ["--max-duration", self.max_dur_var.get().strip() or "6.0"]
        args += ["--min-gap", self.min_gap_var.get().strip() or "0.0"]
        args += ["--chars-per-line", self.chars_var.get().strip() or "42"]
        args += ["--max-lines", self.max_lines_var.get().strip() or "2"]

        if self.translate_var.get():
            args += ["--translate"]
            args += self.build_translation_args()

        return args

    def run_thread(self, script, args):
        script = os.path.join(resource_path(), script)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self.proc = subprocess.Popen(
                [sys.executable, script] + args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True, env=env
            )

            for line in self.proc.stdout:
                line = line.rstrip("\n\r")
                if not line:
                    continue
                clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)
                clean = re.sub(r"\r", "", clean)
                if clean:
                    self.log_queue.put(("log", clean))

            self.proc.wait()
            if self.proc.returncode == 0:
                self.log_queue.put(("status", "Done ✓"))
                self.log_queue.put(("log", "Done ✓"))
            else:
                self.log_queue.put(("status", "Failed ✗"))
                self.log_queue.put(("log", f"Process exited with code {self.proc.returncode}"))
        except Exception as e:
            self.log_queue.put(("log", f"Error: {e}"))
            self.log_queue.put(("status", "Failed ✗"))
        finally:
            self.proc = None
            self.log_queue.put(("finish", None))

    def run(self):
        input_path = self.input_path.get().strip()
        if not input_path:
            kind = "SRT file" if self.is_srt_mode() else "video file"
            messagebox.showwarning("No Input", f"Please select a {kind}.")
            return
        if not os.path.isfile(input_path):
            messagebox.showerror("Error", f"File not found:\n{input_path}")
            return

        self.run_btn.configure(state=tk.DISABLED)
        self.progress.start(10)
        self.clear_log()
        self.set_status("Running...")

        script = "translate_srt.py" if self.is_srt_mode() else "subgen.py"
        args = self.build_args(input_path)
        self.log(f"$ python {script} {' '.join(args)}")
        self.log("")

        t = threading.Thread(target=self.run_thread, args=(script, args), daemon=True)
        t.start()

    def finish(self):
        self.progress.stop()
        self.progress["value"] = 0
        self.run_btn.configure(state=tk.NORMAL)

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    SubGenGUI().start()
