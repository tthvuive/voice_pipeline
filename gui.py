
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

from pipeline import run_speaker_diarization_asr
from export_txt import export_to_txt


APP_TITLE = "Voice Pipeline UI"
BASE_DIR = Path(__file__).resolve().parent
DATA_TRAIN_DIR = BASE_DIR / "data" / "train"
DATA_TEST_DIR = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "models" / "speaker_model.npz"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x700")

        self.selected_label: str | None = None
        self.selected_train_file: str | None = None
        self.selected_test_item: str | None = None  # file name in data/test
        self.last_segments = None  # for export

        self._build_ui()
        self._ensure_dirs()
        self._refresh_all()

    def _ensure_dirs(self) -> None:
        DATA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        DATA_TEST_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)

    # ---------------- UI Layout ----------------
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        # Left column: Data (train labels) + Data TEST
        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, rowspan=2, sticky="nsw")
        left.columnconfigure(0, weight=1)

        # Data (train)
        ttk.Label(left, text="Data", anchor="center").grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.lb_labels = tk.Listbox(left, height=8, exportselection=False)
        self.lb_labels.grid(row=1, column=0, sticky="ew")
        self.lb_labels.bind("<<ListboxSelect>>", self._on_select_label)

        frm_label_btn = ttk.Frame(left)
        frm_label_btn.grid(row=2, column=0, sticky="ew", pady=6)
        frm_label_btn.columnconfigure(0, weight=1)
        frm_label_btn.columnconfigure(1, weight=1)
        ttk.Button(frm_label_btn, text="ADD", command=self._add_label).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(frm_label_btn, text="Delete", command=self._delete_label).grid(row=0, column=1, sticky="ew")

        # Data TEST
        ttk.Label(left, text="Data TEST", anchor="center").grid(row=3, column=0, sticky="ew", pady=(14, 6))
        frm_test_btn = ttk.Frame(left)
        frm_test_btn.grid(row=4, column=0, sticky="ew")
        frm_test_btn.columnconfigure(0, weight=1)
        frm_test_btn.columnconfigure(1, weight=1)
        ttk.Button(frm_test_btn, text="ADD", command=self._add_test_item).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(frm_test_btn, text="Delete", command=self._delete_test_item).grid(row=0, column=1, sticky="ew")

        # Middle column: Person files + test list
        mid = ttk.Frame(self, padding=10)
        mid.grid(row=0, column=1, rowspan=2, sticky="nsw")
        mid.columnconfigure(0, weight=1)

        self.lbl_person = ttk.Label(mid, text="Person: -", anchor="center")
        self.lbl_person.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.lb_train_files = tk.Listbox(mid, height=8, exportselection=False)
        self.lb_train_files.grid(row=1, column=0, sticky="ew")
        self.lb_train_files.bind("<<ListboxSelect>>", self._on_select_train_file)

        frm_train_btn = ttk.Frame(mid)
        frm_train_btn.grid(row=2, column=0, sticky="ew", pady=6)
        frm_train_btn.columnconfigure(0, weight=1)
        frm_train_btn.columnconfigure(1, weight=1)
        ttk.Button(frm_train_btn, text="ADD", command=self._add_train_file).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(frm_train_btn, text="Delete", command=self._delete_train_file).grid(row=0, column=1, sticky="ew")

        self.lbl_testlist = ttk.Label(mid, text="Test items", anchor="center")
        self.lbl_testlist.grid(row=3, column=0, sticky="ew", pady=(14, 6))

        self.lb_test_items = tk.Listbox(mid, height=6, exportselection=False)
        self.lb_test_items.grid(row=4, column=0, sticky="ew")
        self.lb_test_items.bind("<<ListboxSelect>>", self._on_select_test_item)
        self.lb_test_items.bind("<Double-Button-1>", lambda _e: self._run_test())

        # Right top: TRAIN button
        right_top = ttk.Frame(self, padding=10)
        right_top.grid(row=0, column=2, sticky="new")
        right_top.columnconfigure(0, weight=1)

        self.btn_train = ttk.Button(right_top, text="TRAIN", command=self._train_model)
        self.btn_train.grid(row=0, column=0, sticky="w", ipadx=30, ipady=10)

        self.btn_run = ttk.Button(right_top, text="RUN TEST", command=self._run_test)
        self.btn_run.grid(row=0, column=1, sticky="w", padx=(10, 0), ipadx=20, ipady=10)

        # Right bottom: Result + save
        right_bottom = ttk.Frame(self, padding=10)
        right_bottom.grid(row=1, column=2, sticky="nsew")
        right_bottom.columnconfigure(0, weight=1)
        right_bottom.rowconfigure(1, weight=1)

        ttk.Label(right_bottom, text="Result:", anchor="w").grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.txt_result = tk.Text(right_bottom, wrap="word")
        self.txt_result.grid(row=1, column=0, sticky="nsew")
        self.txt_result.insert("1.0", "Chọn Data (train), thêm audio, Train. Sau đó chọn test item và RUN TEST.\n")

        frm_save = ttk.Frame(right_bottom)
        frm_save.grid(row=2, column=0, sticky="e", pady=(8, 0))
        self.btn_save = ttk.Button(frm_save, text="save", command=self._save_result)
        self.btn_save.grid(row=0, column=0, ipadx=20)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew")

    # ---------------- Refresh helpers ----------------
    def _refresh_all(self) -> None:
        self._refresh_labels()
        self._refresh_test_items()
        self._refresh_train_files()

    def _refresh_labels(self) -> None:
        self.lb_labels.delete(0, tk.END)
        labels = sorted([p.name for p in DATA_TRAIN_DIR.iterdir() if p.is_dir()])
        for name in labels:
            self.lb_labels.insert(tk.END, name)

        # keep selection if possible
        if self.selected_label in labels:
            idx = labels.index(self.selected_label)
            self.lb_labels.selection_set(idx)
            self.lb_labels.see(idx)
        else:
            self.selected_label = None
            self.lbl_person.config(text="Person: -")

    def _refresh_train_files(self) -> None:
        self.lb_train_files.delete(0, tk.END)
        if not self.selected_label:
            return
        spk_dir = DATA_TRAIN_DIR / self.selected_label
        files = sorted([p.name for p in spk_dir.glob("*.wav")])
        for f in files:
            self.lb_train_files.insert(tk.END, f)

        if self.selected_train_file in files:
            idx = files.index(self.selected_train_file)
            self.lb_train_files.selection_set(idx)
            self.lb_train_files.see(idx)
        else:
            self.selected_train_file = None

        self.lbl_person.config(text=f"Person: {self.selected_label}")

    def _refresh_test_items(self) -> None:
        self.lb_test_items.delete(0, tk.END)
        items = sorted([p.name for p in DATA_TEST_DIR.glob("*.wav")])
        for it in items:
            self.lb_test_items.insert(tk.END, it)

        if self.selected_test_item in items:
            idx = items.index(self.selected_test_item)
            self.lb_test_items.selection_set(idx)
            self.lb_test_items.see(idx)
        else:
            self.selected_test_item = None

    # ---------------- Event handlers ----------------
    def _on_select_label(self, _evt=None) -> None:
        sel = self.lb_labels.curselection()
        if not sel:
            self.selected_label = None
            self._refresh_train_files()
            return
        self.selected_label = self.lb_labels.get(sel[0])
        self._refresh_train_files()

    def _on_select_train_file(self, _evt=None) -> None:
        sel = self.lb_train_files.curselection()
        self.selected_train_file = self.lb_train_files.get(sel[0]) if sel else None

    def _on_select_test_item(self, _evt=None) -> None:
        sel = self.lb_test_items.curselection()
        self.selected_test_item = self.lb_test_items.get(sel[0]) if sel else None
        self.status.set(f"Selected test: {self.selected_test_item}" if self.selected_test_item else "Ready")

    # ---------------- Data Train actions ----------------
    def _add_label(self) -> None:
        name = simpledialog.askstring("Add Data", "Nhập tên label/person mới:")
        if not name:
            return
        safe = name.strip()
        if not safe:
            return
        target = DATA_TRAIN_DIR / safe
        if target.exists():
            messagebox.showwarning("Tồn tại", f"Label '{safe}' đã tồn tại.")
            return
        target.mkdir(parents=True, exist_ok=True)
        self.selected_label = safe
        self._refresh_labels()
        self._refresh_train_files()
        self._log(f"Đã thêm label train: {safe}")

    def _delete_label(self) -> None:
        if not self.selected_label:
            messagebox.showinfo("Delete", "Chưa chọn label để xoá.")
            return
        label = self.selected_label
        if not messagebox.askyesno("Xác nhận", f"Xoá label '{label}' và toàn bộ audio bên trong?"):
            return
        shutil.rmtree(DATA_TRAIN_DIR / label, ignore_errors=True)
        self.selected_label = None
        self.selected_train_file = None
        self._refresh_labels()
        self._refresh_train_files()
        self._log(f"Đã xoá label train: {label}")

    def _add_train_file(self) -> None:
        if not self.selected_label:
            messagebox.showinfo("ADD", "Hãy chọn 1 label/person ở khối Data trước.")
            return
        filepaths = filedialog.askopenfilenames(
            title="Chọn file .wav để thêm vào train",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not filepaths:
            return
        spk_dir = DATA_TRAIN_DIR / self.selected_label
        spk_dir.mkdir(parents=True, exist_ok=True)
        for fp in filepaths:
            src = Path(fp)
            dst = spk_dir / src.name
            if dst.exists():
                self._log(f"Bỏ qua (đã tồn tại): {dst.name}")
                continue
            shutil.copy2(src, dst)
            self._log(f"Đã thêm train file: {dst.name} -> {self.selected_label}")
        self._refresh_train_files()

    def _delete_train_file(self) -> None:
        if not self.selected_label or not self.selected_train_file:
            messagebox.showinfo("Delete", "Chưa chọn file train để xoá.")
            return
        fp = DATA_TRAIN_DIR / self.selected_label / self.selected_train_file
        if not fp.exists():
            self._log("File không tồn tại trên disk.")
            return
        if not messagebox.askyesno("Xác nhận", f"Xoá file train '{self.selected_train_file}'?"):
            return
        fp.unlink(missing_ok=True)
        self._log(f"Đã xoá train file: {self.selected_train_file}")
        self.selected_train_file = None
        self._refresh_train_files()

    # ---------------- Data TEST actions ----------------
    def _add_test_item(self) -> None:
        filepaths = filedialog.askopenfilenames(
            title="Chọn file .wav để thêm vào Data TEST",
            filetypes=[("WAV audio", "*.wav")],
        )
        if not filepaths:
            return
        for fp in filepaths:
            src = Path(fp)
            dst = DATA_TEST_DIR / src.name
            if dst.exists():
                self._log(f"Bỏ qua (đã tồn tại): {dst.name}")
                continue
            shutil.copy2(src, dst)
            self._log(f"Đã thêm test item: {dst.name}")
        self._refresh_test_items()

    def _delete_test_item(self) -> None:
        if not self.selected_test_item:
            messagebox.showinfo("Delete", "Chưa chọn test item để xoá.")
            return
        fp = DATA_TEST_DIR / self.selected_test_item
        if not fp.exists():
            self._log("Test item không tồn tại trên disk.")
            return
        if not messagebox.askyesno("Xác nhận", f"Xoá test item '{self.selected_test_item}'?"):
            return
        fp.unlink(missing_ok=True)
        self._log(f"Đã xoá test item: {self.selected_test_item}")
        self.selected_test_item = None
        self._refresh_test_items()

    # ---------------- Train / Test ----------------
    def _train_model(self) -> None:
        # Run train in background thread to keep UI responsive.
        def job():
            try:
                self._set_busy(True, "Training...")
                self._log("Bắt đầu TRAIN...")
                # call existing script logic by importing
                import train_classifier  # noqa: F401  (side-effect training)
                self._log("TRAIN xong. Model lưu tại models/speaker_model.npz")
                self.status.set("Train completed.")
            except Exception as e:
                messagebox.showerror("Train error", str(e))
                self._log(f"Train error: {e}")
                self.status.set("Train failed.")
            finally:
                self._set_busy(False, "Ready")

        threading.Thread(target=job, daemon=True).start()

    def _run_test(self) -> None:
        if not self.selected_test_item:
            messagebox.showinfo("RUN TEST", "Hãy chọn 1 test item trong danh sách Test items.")
            return
        if not MODEL_PATH.exists():
            messagebox.showwarning("Chưa có model", "Chưa có model. Hãy bấm TRAIN trước.")
            return

        wav_path = str(DATA_TEST_DIR / self.selected_test_item)

        def job():
            try:
                self._set_busy(True, "Running test...")
                self._log(f"RUN TEST: {self.selected_test_item}")
                segments, pretty = run_speaker_diarization_asr(
                    wav_path=wav_path,
                    model_path=str(MODEL_PATH),
                    segment_len=1.5,
                )
                self.last_segments = segments
                self._set_result(pretty)
                self.status.set(f"Done: {self.selected_test_item}")
            except Exception as e:
                messagebox.showerror("Run error", str(e))
                self._log(f"Run error: {e}")
                self.status.set("Run failed.")
            finally:
                self._set_busy(False, "Ready")

        threading.Thread(target=job, daemon=True).start()

    # ---------------- Save ----------------
    def _save_result(self) -> None:
        if not self.selected_test_item:
            messagebox.showinfo("Save", "Hãy chọn 1 test item để đặt tên file output.")
            return

        base_name = Path(self.selected_test_item).stem
        default_name = f"{base_name}_result.txt"

        out_path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            initialfile=default_name,
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
        )
        if not out_path:
            return

        # If we have structured segments, export with timestamps; else export raw text box.
        try:
            if self.last_segments:
                export_to_txt(self.last_segments, output_file=out_path)
            else:
                txt = self.txt_result.get("1.0", "end-1c")
                Path(out_path).write_text(txt, encoding="utf-8")
            self._log(f"Saved: {out_path}")
            self.status.set(f"Saved to {out_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"Save error: {e}")

    # ---------------- Utility ----------------
    def _set_result(self, text: str) -> None:
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", text)

    def _log(self, msg: str) -> None:
        self.txt_result.insert(tk.END, msg + "\n")
        self.txt_result.see(tk.END)

    def _set_busy(self, busy: bool, status: str) -> None:
        def ui():
            self.btn_train.config(state=("disabled" if busy else "normal"))
            self.btn_run.config(state=("disabled" if busy else "normal"))
            self.btn_save.config(state=("disabled" if busy else "normal"))
            self.status.set(status)
        self.after(0, ui)


def main() -> None:
    # Tk themed widgets
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
