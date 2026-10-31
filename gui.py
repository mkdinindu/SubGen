#!/usr/bin/env python3
"""GUI frontend for SubGen — subtitle generator with Whisper + Ollama."""

import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


LANGUAGES = [
    "", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja",
    "ko", "zh", "ar", "hi", "tr", "pl", "sv", "da", "fi", "no",
    "cs", "hu", "ro", "el", "he", "th", "vi", "id", "ms"
]


def resource_path():
    return os.path.dirname(os.path.abspath(__file__))


class SubGenGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SubGen — Subtitle Generator")
        self.root.geometry("720x660")
        self.root.resizable(True, True)
        self.proc = None
        self.trans_frame = None
        self.build_widgets()

    def build_widgets(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(6, weight=1)

        # ── Video file ──
        ttk.Label(main, text="Video File:", font=("", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 2))
        row1 = ttk.Frame(main)
        row1.grid(row=1, column=0, sticky=tk.EW, pady=(0, 8))
        row1.columnconfigure(0, weight=1)
        self.video_path = tk.StringVar()
        ttk.Entry(row1, textvariable=self.video_path).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(row1, text="Browse", command=self.browse_video).grid(row=0, column=1, padx=(6, 0))

        # ── Whisper settings ──
        self.model_var = tk.StringVar(value="base")
        self.lang_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="")

        opts_frame = ttk.LabelFrame(main, text="Whisper Settings", padding=8)
        opts_frame.grid(row=2, column=0, sticky=tk.EW, pady=(0, 8))
        opts_frame.columnconfigure(1, weight=1)

        ttk.Label(opts_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(opts_frame, textvariable=self.model_var,
                      values=["tiny", "base", "small", "medium", "large"],
                      state="readonly", width=14).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(opts_frame, text="Audio Language:").grid(row=0, column=2, sticky=tk.W, padx=(12, 6))
        lang_cb = ttk.Combobox(opts_frame, textvariable=self.lang_var,
                               values=LANGUAGES, state="normal", width=8)
        lang_cb.grid(row=0, column=3, sticky=tk.W)

        ttk.Label(opts_frame, text="Output:").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        ttk.Entry(opts_frame, textvariable=self.output_var).grid(
            row=1, column=1, columnspan=3, sticky=tk.EW, pady=(6, 0))

        # ── Timing ──
        timing_frame = ttk.LabelFrame(main, text="Timing", padding=8)
        timing_frame.grid(row=3, column=0, sticky=tk.EW, pady=(0, 8))

        self.delay_var = tk.StringVar(value="0.0")
        self.min_dur_var = tk.StringVar(value="1.0")
        self.max_dur_var = tk.StringVar(value="6.0")
        self.chars_var = tk.StringVar(value="42")

        ttk.Label(timing_frame, text="Delay (s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(timing_frame, textvariable=self.delay_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(timing_frame, text="Min Dur (s):").grid(row=0, column=2, sticky=tk.W, padx=(12, 4))
        ttk.Entry(timing_frame, textvariable=self.min_dur_var, width=8).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(timing_frame, text="Max Dur (s):").grid(row=1, column=0, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        ttk.Entry(timing_frame, textvariable=self.max_dur_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=(6, 0))
        ttk.Label(timing_frame, text="Chars/Line:").grid(row=1, column=2, sticky=tk.W, padx=(12, 4), pady=(6, 0))
        ttk.Entry(timing_frame, textvariable=self.chars_var, width=8).grid(row=1, column=3, sticky=tk.W, pady=(6, 0))

        # ── Translation ──
        self.translate_var = tk.BooleanVar(value=False)
        self.trans_frame = ttk.LabelFrame(main, text="Translation (Ollama)", padding=8)
        self.trans_frame.grid(row=4, column=0, sticky=tk.EW, pady=(0, 8))
        self.trans_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(self.trans_frame, text="Enable Translation",
                        variable=self.translate_var,
                        command=self.toggle_translate).grid(row=0, column=0, columnspan=5, sticky=tk.W)

        self.src_lang_var = tk.StringVar(value="English")
        self.src_code_var = tk.StringVar(value="en")
        self.tgt_lang_var = tk.StringVar(value="Spanish")
        self.tgt_code_var = tk.StringVar(value="es")
        self.ollama_var = tk.StringVar(value="translategemma:4b")

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

        self.toggle_translate()

        # ── Progress & Log ──
        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky=tk.EW, pady=(0, 6))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).grid(row=6, column=0, sticky=tk.W, pady=(0, 4))
        main.rowconfigure(7, weight=1)

        log_frame = ttk.LabelFrame(main, text="Output Log", padding=4)
        log_frame.grid(row=7, column=0, sticky=tk.NSEW, pady=(0, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED,
                                bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9), relief=tk.FLAT)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        # ── Buttons ──
        btn_row = ttk.Frame(main)
        btn_row.grid(row=8, column=0, sticky=tk.EW)
        self.run_btn = ttk.Button(btn_row, text="Generate Subtitles", command=self.run, width=24)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)

    # ── helpers ──

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            if not self.output_var.get():
                self.output_var.set(os.path.splitext(path)[0] + ".srt")

    def toggle_translate(self):
        state = tk.NORMAL if self.translate_var.get() else tk.DISABLED
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
        self.root.update_idletasks()

    # ── run pipeline ──

    def build_args(self, video):
        args = [video]
        model = self.model_var.get()
        if model != "base":
            args += ["--whisper-model", model]
        lang = self.lang_var.get().strip()
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
        args += ["--chars-per-line", self.chars_var.get().strip() or "42"]

        if self.translate_var.get():
            args += ["--translate"]
            args += ["--source-lang", self.src_lang_var.get().strip()]
            args += ["--source-code", self.src_code_var.get().strip()]
            args += ["--target-lang", self.tgt_lang_var.get().strip()]
            args += ["--target-code", self.tgt_code_var.get().strip()]
            args += ["--ollama-model", self.ollama_var.get().strip()]

        return args

    def run_thread(self, args):
        script = os.path.join(resource_path(), "subgen.py")
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
                    self.log(clean)

            self.proc.wait()
            if self.proc.returncode == 0:
                self.set_status("Done ✓")
                self.log("Done ✓")
            else:
                self.set_status("Failed ✗")
                self.log(f"Process exited with code {self.proc.returncode}")
        except Exception as e:
            self.log(f"Error: {e}")
            self.set_status("Failed ✗")
        finally:
            self.proc = None
            self.root.after(0, self.finish)

    def run(self):
        video = self.video_path.get().strip()
        if not video:
            messagebox.showwarning("No Video", "Please select a video file.")
            return
        if not os.path.isfile(video):
            messagebox.showerror("Error", f"File not found:\n{video}")
            return

        if self.translate_var.get() and not shutil.which("ollama"):
            messagebox.showerror("Error", "ollama not found.\nInstall from https://ollama.com")
            return

        self.run_btn.configure(state=tk.DISABLED)
        self.progress.start(10)
        self.clear_log()
        self.set_status("Running...")

        args = self.build_args(video)
        self.log(f"$ python subgen.py {' '.join(args)}")
        self.log("")

        t = threading.Thread(target=self.run_thread, args=(args,), daemon=True)
        t.start()

    def finish(self):
        self.progress.stop()
        self.progress["value"] = 0
        self.run_btn.configure(state=tk.NORMAL)

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    SubGenGUI().start()
