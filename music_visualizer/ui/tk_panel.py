"""Tkinter settings panel (OpenCV display fallback when GL/ImGui UI is unavailable)."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

from music_visualizer.config import Config
from music_visualizer.ui.controller import UIController
from music_visualizer.ui.persistence import export_settings, import_settings, save_defaults
from music_visualizer.audio.routing import parse_route_key, routes_file_path, save_routes_file, set_route_field
from music_visualizer.analysis.vocab_lists import parse_vocab_key, save_all_vocab_lists, set_vocab_field
from music_visualizer.ui.settings_schema import (
    ParamKind,
    ParamSpec,
    config_value_for_ui,
    ui_value_to_config,
)


class _CollapsibleSection(ttk.Frame):
    """Settings group that can expand/collapse; state is owned by UIController."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        collapsed: bool,
        on_toggle: Any,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._collapsed = collapsed
        self._on_toggle = on_toggle
        header = ttk.Frame(self)
        header.pack(fill=tk.X)
        sign = "\u25b8 " if collapsed else "\u25be "
        self._btn = ttk.Button(
            header,
            text=sign + title,
            command=self._toggle,
            style="Toolbutton",
        )
        self._btn.pack(fill=tk.X, anchor=tk.W)
        self.body = ttk.Frame(self)
        if not collapsed:
            self.body.pack(fill=tk.X, padx=4, pady=2)
        self.pack(fill=tk.X, padx=6, pady=2)

    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        sign = "\u25b8 " if self._collapsed else "\u25be "
        self._btn.configure(text=sign + self._title)
        if self._collapsed:
            self.body.pack_forget()
        else:
            self.body.pack(fill=tk.X, padx=4, pady=2)
        self._on_toggle(self._collapsed)


class TkSettingsPanel:
    """Scrollable settings window toggled with F1 on the OpenCV display path."""

    def __init__(self, controller: UIController) -> None:
        self._controller = controller
        self._root: Optional[tk.Toplevel] = None
        self._vars: Dict[str, Any] = {}
        self._visible = False

    @property
    def visible(self) -> bool:
        return self._visible

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def show(self) -> None:
        if self._root is not None:
            self._root.deiconify()
            self._root.lift()
            self._visible = True
            return
        self._build()
        self._visible = True

    def hide(self) -> None:
        if self._root is not None:
            self._root.withdraw()
        self._visible = False

    def _build(self) -> None:
        cfg = self._controller._ctx.cfg
        root = tk.Toplevel()
        root.title("Music Visualizer Settings")
        root.geometry("560x720")
        root.attributes("-alpha", 0.92)
        root.protocol("WM_DELETE_WINDOW", self.hide)

        bar = ttk.Frame(root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="Save Defaults", command=self._save_defaults).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Export", command=self._export).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Import", command=self._import).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Restart", command=self._restart).pack(side=tk.LEFT, padx=2)

        if self._controller._ctx.restart.pending:
            ttk.Label(
                root,
                text="Some settings need a restart — click Restart when ready.",
                wraplength=520,
            ).pack(fill=tk.X, padx=8, pady=4)

        canvas = tk.Canvas(root, highlightthickness=0)
        scroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        style = ttk.Style(root)
        style.configure("Restart.TLabel", foreground="#FFB85A")

        current_group = None
        group_frame = inner
        for spec in self._controller.specs:
            if spec.group != current_group:
                current_group = spec.group
                section = _CollapsibleSection(
                    inner,
                    spec.group,
                    self._controller.is_group_collapsed(spec.group),
                    lambda collapsed, g=spec.group: self._controller.set_group_collapsed(
                        g, collapsed
                    ),
                )
                group_frame = section.body
            self._add_widget(group_frame, spec, config_value_for_ui(cfg, spec))

        self._root = root

    def _add_widget(self, parent: ttk.Frame, spec: ParamSpec, value: Any) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=4)
        lbl_style = "Restart.TLabel" if spec.requires_restart else "TLabel"
        lbl = ttk.Label(row, text=spec.label, wraplength=520, anchor=tk.W, justify=tk.LEFT, style=lbl_style)
        lbl.pack(fill=tk.X, anchor=tk.W)
        if spec.tooltip:
            _tooltip(lbl, spec.tooltip)

        ctrl = ttk.Frame(row)
        ctrl.pack(fill=tk.X, pady=(2, 0))
        widget: tk.Widget

        if spec.kind == ParamKind.BOOL:
            var = tk.BooleanVar(value=bool(value))
            widget = ttk.Checkbutton(ctrl, variable=var, command=lambda: self._on_change(spec, var.get()))
            widget.pack(anchor=tk.W)
            self._vars[spec.key] = var
        elif spec.kind == ParamKind.ENUM:
            var = tk.StringVar(value=str(value))
            widget = ttk.Combobox(ctrl, textvariable=var, values=list(spec.options), state="readonly")
            widget.pack(fill=tk.X)
            widget.bind("<<ComboboxSelected>>", lambda _e: self._on_change(spec, var.get()))
            self._vars[spec.key] = var
        elif spec.kind == ParamKind.TEXT:
            var = tk.StringVar(value=str(value or ""))
            widget = ttk.Entry(ctrl, textvariable=var)
            widget.pack(fill=tk.X)
            widget.bind("<Return>", lambda _e: self._on_change(spec, var.get()))
            widget.bind("<FocusOut>", lambda _e: self._on_change(spec, var.get()))
            self._vars[spec.key] = var
        elif spec.kind == ParamKind.INT:
            var = tk.IntVar(value=int(value))
            widget = ttk.Spinbox(
                ctrl, from_=int(spec.min_val), to=int(spec.max_val),
                textvariable=var,
                command=lambda: self._on_change(spec, var.get()),
            )
            widget.pack(fill=tk.X)
            widget.bind("<Return>", lambda _e: self._on_change(spec, var.get()))
            widget.bind("<FocusOut>", lambda _e: self._on_change(spec, var.get()))
            self._vars[spec.key] = var
        else:
            var = tk.DoubleVar(value=float(value))
            widget = ttk.Spinbox(
                ctrl, from_=spec.min_val, to=spec.max_val, increment=spec.step or 0.01,
                textvariable=var,
                command=lambda: self._on_change(spec, var.get()),
            )
            widget.pack(fill=tk.X)
            widget.bind("<Return>", lambda _e: self._on_change(spec, var.get()))
            widget.bind("<FocusOut>", lambda _e: self._on_change(spec, var.get()))
            self._vars[spec.key] = var

        for w in (row, lbl, ctrl, widget):
            w.bind("<Double-Button-1>", lambda _e, s=spec: self._reset_to_default(s))
        if spec.kind not in (ParamKind.TEXT, ParamKind.BOOL):
            widget.bind("<Button-3>", lambda _e, w=widget: w.focus_set())

    def _reset_to_default(self, spec: ParamSpec) -> None:
        defaults = Config()
        val = config_value_for_ui(defaults, spec)
        var = self._vars.get(spec.key)
        if var is None:
            return
        if isinstance(var, tk.BooleanVar):
            var.set(bool(val))
        elif isinstance(var, tk.IntVar):
            var.set(int(val))
        elif isinstance(var, tk.DoubleVar):
            var.set(float(val))
        else:
            var.set(str(val or ""))
        self._controller.reset_key(spec.key)

    def _on_change(self, spec: ParamSpec, raw: Any) -> None:
        self._controller.apply_key(spec.key, raw)

    def _sync_vars_to_config(self) -> None:
        cfg = self._controller._ctx.cfg
        for spec in self._controller.specs:
            var = self._vars.get(spec.key)
            if var is None:
                continue
            val = ui_value_to_config(spec, var.get())
            parsed = parse_route_key(spec.key)
            if parsed is not None:
                idx, field_name = parsed
                set_route_field(cfg, idx, field_name, val)
            else:
                parsed_vocab = parse_vocab_key(spec.key)
                if parsed_vocab is not None:
                    idx, field_name = parsed_vocab
                    set_vocab_field(cfg, idx, field_name, val)
                else:
                    setattr(cfg, spec.key, val)

    def _save_defaults(self) -> None:
        self._sync_vars_to_config()
        path = save_defaults(self._controller._ctx.cfg)
        try:
            save_routes_file(
                routes_file_path(self._controller._ctx.cfg),
                self._controller._ctx.cfg.react_routes or [],
            )
        except OSError:
            pass
        try:
            save_all_vocab_lists(self._controller._ctx.cfg)
        except OSError:
            pass
        messagebox.showinfo("Saved", f"Defaults saved to\n{path}", parent=self._root)

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")], parent=self._root,
        )
        if path:
            export_settings(self._controller._ctx.cfg, __import__("pathlib").Path(path))

    def _import(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], parent=self._root)
        if path:
            data = import_settings(__import__("pathlib").Path(path))
            self._controller.import_dict(data)

    def _restart(self) -> None:
        self._controller._ctx.restart.trigger()


def _tooltip(widget: tk.Widget, text: str) -> None:
    tip: dict[str, Optional[tk.Toplevel]] = {"win": None}

    def enter(_event):
        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{widget.winfo_rootx()+20}+{widget.winfo_rooty()+20}")
        ttk.Label(tw, text=text, wraplength=280, padding=4).pack()
        tip["win"] = tw

    def leave(_event):
        if tip["win"] is not None:
            tip["win"].destroy()
            tip["win"] = None

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)
