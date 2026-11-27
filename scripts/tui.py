"""
PicoCode v0.4.0 — Claude-Code / OpenCode-Grade 24-bit TrueColor Terminal UI.

Uses `rich` and `prompt_toolkit` when installed (with graceful stdlib fallback)
to deliver:
  - Rounded 2-column welcome header card with gradient PICOCODE logo
  - Interactive `❯` prompt with live `/` slash-command and `@file` dropdown autocomplete,
    persistent history, and a bottom status toolbar (model · git branch · ctx% · mode)
  - Live animated braille spinner during startup, thinking, and tool execution
  - Real-time syntax-highlighted Markdown streaming (`rich.live.Live` + `Markdown`)
  - Claude-Code-style `● Tool(args)` / `⎿ summary` tree cards & colorized diff panels
  - Interactive `[y] Run / [a] Always / [n] Deny` command approval panel

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import html
import os
import pathlib
import sys
import time
from typing import Any, Callable

# Ensure UTF-8 + Windows VT100 TrueColor sequences work cleanly in PowerShell & Windows Terminal
def init_windows_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
                h = kernel32.GetStdHandle(handle_id)
                mode = ctypes.c_uint32()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass


try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.markup import escape as rich_escape
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    def rich_escape(s: str) -> str:  # type: ignore[misc]
        return s

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PTStyle
    HAS_PT = True
except ImportError:
    HAS_PT = False


SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/project", "Pick, switch, or create a workspace folder (/project my-app)"),
    ("/trust", "Manage trusted folders (/trust | /trust add | /trust remove)"),
    ("/files", "Browse files in the active project (/files [subdir])"),
    ("/add", "Pin a file into persistent chat context (/add file.py · Tab)"),
    ("/drop", "Unpin files from context (/drop [file.py])"),
    ("/model", "List installed models or hot-swap live (/model inkling2-mini)"),
    ("/plan", "Toggle read-only Plan & Review Mode"),
    ("/diff", "Show working-tree Git diff summary"),
    ("/undo", "Revert all file edits from the last turn"),
    ("/status", "Show active model, project, trust, context, and Git status"),
    ("/yes", "Toggle auto-approval for safe shell commands"),
    ("/clear", "Reset chat history and clear screen"),
    ("/help", "Show command list and shortcuts"),
    ("/exit", "Exit PicoCode Terminal Agent"),
]

TOOL_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    "project_overview": ("Overview", "#38bdf8"),
    "repo_map": ("RepoMap", "#38bdf8"),
    "list_files": ("List", "#38bdf8"),
    "read_file": ("Read", "#60a5fa"),
    "search_code": ("Search", "#818cf8"),
    "git_status": ("GitStatus", "#a78bfa"),
    "git_diff": ("GitDiff", "#a78bfa"),
    "edit_file": ("Edit", "#34d399"),
    "write_file": ("Write", "#34d399"),
    "run_command": ("Shell", "#fbbf24"),
}


def format_tool_arg_preview(name: str, args: dict[str, Any]) -> str:
    if name == "read_file":
        p = str(args.get("path") or "")
        s, e = args.get("start_line"), args.get("end_line")
        rng = f" · L{s}-{e}" if (s or e) else ""
        return f"{p}{rng}"
    if name in ("edit_file", "write_file"):
        return str(args.get("path") or "")
    if name == "search_code":
        return repr(str(args.get("query") or ""))
    if name == "run_command":
        return str(args.get("command") or "")[:90]
    if name == "list_files":
        return str(args.get("subdir") or ".")
    if name == "repo_map":
        sub = str(args.get("subdir") or "").strip()
        return sub or "workspace"
    if name == "git_diff":
        p = str(args.get("path") or "").strip()
        st = " --staged" if args.get("staged") else ""
        return (p or "all") + st
    return ""


if HAS_PT:
    class PicoCompleter(Completer):
        """Interactive dropdown autocomplete for `/`, `/model`, `/project`, `/add`, `/drop`, and `@file`."""

        def __init__(
            self,
            get_files_fn: Callable[[], list[str]],
            get_models_fn: Callable[[], list[dict[str, Any]]] | None = None,
            get_projects_fn: Callable[[], list[dict[str, Any]]] | None = None,
            get_pinned_fn: Callable[[], list[str]] | None = None,
        ) -> None:
            self._get_files = get_files_fn
            self._get_models = get_models_fn or (lambda: [])
            self._get_projects = get_projects_fn or (lambda: [])
            self._get_pinned = get_pinned_fn or (lambda: [])

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            low = text.lower()
            # 1) `/model <id>` sub-completion
            if low.startswith("/model "):
                partial = text[7:].strip().lower()
                for m in self._get_models():
                    mid = str(m.get("id") or "")
                    if partial in mid.lower():
                        tag = "✓ installed" if m.get("installed") else "missing"
                        yield Completion(
                            mid,
                            start_position=-len(text[7:]),
                            display=mid,
                            display_meta=f"{tag} · {m.get('short', '')}",
                        )
                return
            # 2) `/project <name>` or `/cd <name>` sub-completion
            for pref in ("/project ", "/cd "):
                if low.startswith(pref):
                    raw_arg = text[len(pref):]
                    partial = raw_arg.strip().lower()
                    for p in self._get_projects():
                        pname = str(p.get("name") or "")
                        if partial in pname.lower():
                            meta = ("✓ trusted" if p.get("trusted") else "new") + (" · git" if p.get("is_git") else "")
                            yield Completion(
                                pname,
                                start_position=-len(raw_arg),
                                display=pname,
                                display_meta=meta,
                            )
                    return
            # 3) `/add <file>` sub-completion
            if low.startswith("/add "):
                raw_arg = text[5:]
                ql = raw_arg.strip().lower()
                files = self._get_files()
                matches = [f for f in files if ql in f.lower()]
                matches.sort(key=lambda f: (not f.lower().startswith(ql), len(f)))
                for f in matches[:18]:
                    yield Completion(f, start_position=-len(raw_arg), display=f, display_meta="pin to context")
                return
            # 4) `/drop <file>` sub-completion
            if low.startswith("/drop "):
                raw_arg = text[6:]
                ql = raw_arg.strip().lower()
                for f in self._get_pinned():
                    if ql in f.lower():
                        yield Completion(f, start_position=-len(raw_arg), display=f, display_meta="unpin file")
                return
            # 5) Slash commands at start of line
            if text.startswith("/") and " " not in text:
                q = low
                for cmd, desc in SLASH_COMMANDS:
                    if cmd.startswith(q):
                        yield Completion(cmd, start_position=-len(text), display=cmd, display_meta=desc)
                return
            # 6) @file reference anywhere in prompt
            at_pos = text.rfind("@")
            if at_pos != -1 and (at_pos == 0 or text[at_pos - 1].isspace()):
                partial = text[at_pos + 1:]
                if " " not in partial:
                    ql = partial.lower()
                    files = self._get_files()
                    matches = [f for f in files if ql in f.lower()]
                    matches.sort(key=lambda f: (not f.lower().startswith(ql), len(f)))
                    for f in matches[:18]:
                        yield Completion(
                            f,
                            start_position=-len(partial),
                            display=f"@{f}",
                            display_meta="file context",
                        )


class TerminalUI:
    """Rich + prompt_toolkit TUI facade with clean stdlib fallback."""

    def __init__(
        self,
        get_files_fn: Callable[[], list[str]] | None = None,
        get_models_fn: Callable[[], list[dict[str, Any]]] | None = None,
        get_projects_fn: Callable[[], list[dict[str, Any]]] | None = None,
        get_pinned_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        init_windows_console()
        self.rich_enabled = HAS_RICH
        self.console = Console(highlight=False) if HAS_RICH else None
        self._get_files = get_files_fn or (lambda: [])
        self._get_models = get_models_fn or (lambda: [])
        self._get_projects = get_projects_fn or (lambda: [])
        self._get_pinned = get_pinned_fn or (lambda: [])
        self._session = None
        if HAS_PT and sys.stdin.isatty():
            hist_path = pathlib.Path.home() / ".picocode_agent_history"
            style = PTStyle.from_dict({
                "prompt": "bold #38bdf8",
                "rprompt": "#64748b",
                "bottom-toolbar": "bg:#0f172a #94a3b8",
                "bottom-toolbar.key": "bold #38bdf8 bg:#0f172a",
                "bottom-toolbar.ok": "bold #22c55e bg:#0f172a",
                "bottom-toolbar.warn": "bold #f59e0b bg:#0f172a",
                "bottom-toolbar.sep": "#334155 bg:#0f172a",
                "completion-menu.completion": "bg:#111827 #cbd5e1",
                "completion-menu.completion.current": "bg:#1e3a5f #ffffff bold",
                "completion-menu.meta.completion": "bg:#0b1120 #64748b",
                "completion-menu.meta.completion.current": "bg:#1e3a5f #93c5fd",
            })
            self._session = PromptSession(
                history=FileHistory(str(hist_path)),
                completer=PicoCompleter(self._get_files, self._get_models, self._get_projects, self._get_pinned),
                complete_while_typing=True,
                style=style,
            )

    def clear(self) -> None:
        if self.console:
            self.console.clear()
        else:
            os.system("cls" if os.name == "nt" else "clear")

    def print_welcome(
        self,
        *,
        version: str,
        model: str,
        workspace_name: str,
        workspace_path: str,
        git_branch: str,
        git_changes: int,
        plan_only: bool,
        auto_approve: bool,
        trusted: bool = True,
        ctx_limit: int = 32768,
    ) -> None:
        if not self.console:
            print(f"PicoCode {version} | {model} | {workspace_path}")
            return

        logo = Text()
        logo.append("  █▀█ █ █▀▀ █▀█\n", style="bold #38bdf8")
        logo.append("  █▀▀ █ █▄▄ █▄█\n", style="bold #60a5fa")
        logo.append("  █▀▀ █▀█ █▀▄ █▀▀\n", style="bold #818cf8")
        logo.append("  █▄▄ █▄█ █▄▀ ██▄\n", style="bold #a78bfa")
        logo.append(f"\n  v{version}", style="bold #22c55e")
        logo.append(" · Local Agent", style="#64748b")

        meta = Table.grid(padding=(0, 1))
        meta.add_column(style="bold #64748b", justify="right", no_wrap=True)
        meta.add_column(style="#e2e8f0")

        mode_str = "[bold #f59e0b]◐ PLAN (read-only)[/]" if plan_only else "[bold #22c55e]● AGENT (16 steps)[/]"
        if auto_approve:
            mode_str += "  [bold #38bdf8]⚡ --yes[/]"
        trust_tag = "[#22c55e]🔒 Trusted[/]" if trusted else "[#f59e0b]⚠ Read-Only[/]"
        git_str = f"[#a78bfa]⎇ {git_branch}[/]" if git_branch else "[#64748b]no git[/]"
        if git_branch:
            git_str += f"  [#f59e0b]({git_changes} changed)[/]" if git_changes else "  [#22c55e](clean)[/]"

        meta.add_row("Model", f"[bold #38bdf8]{rich_escape(model)}[/]  [#64748b]({ctx_limit // 1024}K ctx)[/]")
        meta.add_row("Project", f"[bold #f8fafc]{rich_escape(workspace_name)}[/]  {trust_tag}  [#64748b]{rich_escape(workspace_path)}[/]")
        meta.add_row("Git", git_str)
        meta.add_row("Mode", mode_str)
        meta.add_row(
            "Commands",
            "[#38bdf8]/project[/]  [#38bdf8]/model[/]  [#38bdf8]/files[/]  [#38bdf8]/add[/]  [#38bdf8]/plan[/]  [#38bdf8]/undo[/]  [#64748b]· Tab to complete[/]",
        )

        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(width=22)
        grid.add_column(ratio=1)
        grid.add_row(logo, meta)

        panel = Panel(
            grid,
            border_style="#2563eb",
            box=box.ROUNDED,
            padding=(0, 1),
            title="[bold #38bdf8]✦ PicoCode Terminal Agent[/]",
            title_align="left",
            subtitle="[#64748b]Apache-2.0 · 100% Local[/]",
            subtitle_align="right",
        )
        self.console.print()
        self.console.print(panel)

    def read_prompt(
        self,
        *,
        model: str,
        workspace_name: str = "",
        git_branch: str = "",
        ctx_tokens: int = 0,
        ctx_limit: int = 32768,
        plan_only: bool = False,
        auto_approve: bool = False,
        pinned_count: int = 0,
    ) -> str:
        pct = min(100, int(round((ctx_tokens / max(1, ctx_limit)) * 100)))
        k_str = f"{ctx_tokens / 1000:.1f}k" if ctx_tokens >= 1000 else str(ctx_tokens)

        if self.console:
            w = max(40, min(self.console.width, 120))
            self.console.print("─" * w, style="#1e293b")

        if self._session is not None:
            mode_cls = "bottom-toolbar.warn" if plan_only else "bottom-toolbar.ok"
            mode_lbl = "PLAN" if plan_only else "AGENT"
            yes_lbl = " · YES" if auto_approve else ""
            safe_model = html.escape(model)
            safe_ws = html.escape(workspace_name)
            safe_br = html.escape(git_branch)
            proj_part = f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  📁 {safe_ws}" if safe_ws else ""
            branch_part = f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  ⎇ {safe_br}" if safe_br else ""
            pin_part = f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  <bottom-toolbar.key>📎 {pinned_count}</bottom-toolbar.key>" if pinned_count else ""
            ctx_cls = "bottom-toolbar.warn" if pct >= 65 else "bottom-toolbar.key"

            def _toolbar():
                return HTML(
                    f" <{mode_cls}>● {mode_lbl}{yes_lbl}</{mode_cls}>"
                    f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  <bottom-toolbar.key>{safe_model}</bottom-toolbar.key>"
                    f"{proj_part}{branch_part}{pin_part}"
                    f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  ctx <{ctx_cls}>{k_str} ({pct}%)</{ctx_cls}>"
                    f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  /project · /model · /add"
                )

            rprompt = HTML("<rprompt>Ctrl+C exit</rprompt>")
            return self._session.prompt(
                HTML("<prompt>❯ </prompt>"),
                rprompt=rprompt,
                bottom_toolbar=_toolbar,
            ).strip()

        tag = "[PLAN]" if plan_only else "[AGENT]"
        return input(f"{tag} ❯ ").strip()

    def print_info(self, msg: str, style: str = "#94a3b8") -> None:
        if self.console:
            self.console.print(f"  {msg}", style=style)
        else:
            print(f"  {msg}")

    def print_ok(self, msg: str) -> None:
        if self.console:
            self.console.print(f"  [bold #22c55e]✓[/] [#e2e8f0]{rich_escape(msg)}[/]")
        else:
            print(f"  ✓ {msg}")

    def print_warn(self, msg: str) -> None:
        if self.console:
            self.console.print(f"  [bold #f59e0b]▲[/] [#fde68a]{rich_escape(msg)}[/]")
        else:
            print(f"  ▲ {msg}")

    def print_error(self, msg: str) -> None:
        if self.console:
            self.console.print(f"  [bold #ef4444]✗[/] [#fca5a5]{rich_escape(msg)}[/]")
        else:
            print(f"  ✗ {msg}")

    def print_help(self) -> None:
        if not self.console:
            for c, d in SLASH_COMMANDS:
                print(f"  {c:<10} {d}")
            return
        tbl = Table(box=box.ROUNDED, border_style="#334155", show_header=False, padding=(0, 1))
        tbl.add_column("Cmd", style="bold #38bdf8", no_wrap=True)
        tbl.add_column("Desc", style="#cbd5e1")
        for c, d in SLASH_COMMANDS:
            tbl.add_row(c, d)
        tbl.add_row("@file", "Inline file content directly into your prompt (Tab to pick)")
        self.console.print(Panel(tbl, title="[bold #38bdf8]Commands[/]", border_style="#334155", box=box.ROUNDED))

    def print_models_table(self, models: list[dict[str, Any]], active_model: str) -> None:
        if not self.console:
            for m in models:
                st = "*" if m["id"] == active_model else ("OK" if m["installed"] else "MISSING")
                print(f"  [{st}] {m['id']} ({m['ctx_limit'] // 1024}K) - {m['label']}")
            return
        tbl = Table(box=box.ROUNDED, border_style="#334155", padding=(0, 1))
        tbl.add_column("Model ID", style="bold #38bdf8", no_wrap=True)
        tbl.add_column("Status", no_wrap=True)
        tbl.add_column("Context", justify="right", style="#94a3b8", no_wrap=True)
        tbl.add_column("Profile", style="#cbd5e1")
        for m in models:
            if m["id"] == active_model:
                st = "[bold #22c55e]● Active[/]"
            elif m["installed"]:
                st = "[#38bdf8]✓ Installed[/]"
            else:
                st = "[#64748b]✗ Missing[/]"
            tbl.add_row(m["id"], st, f"{m['ctx_limit'] // 1024}K", m.get("short") or m.get("label", ""))
        self.console.print(
            Panel(
                tbl,
                title="[bold #38bdf8]✦ Local Coding Models[/]",
                subtitle="[#64748b]Live switch: /model inkling2-mini  or  /model qwen3-coder-30b[/]",
                border_style="#334155",
                box=box.ROUNDED,
            )
        )

    def prompt_trust_folder(self, folder_path: str) -> str:
        """Claude-Code-style Folder Trust gate. Returns 'trust', 'plan', or 'deny'."""
        if self.console:
            body = Text()
            body.append(f"{folder_path}\n\n", style="bold #f8fafc")
            body.append(
                "PicoCode can read and edit files in this folder, and run shell commands with your approval.\n"
                "Do you trust the files and commands in this folder?\n\n",
                style="#cbd5e1",
            )
            body.append("  [1] ", style="bold #22c55e")
            body.append("Yes, trust this folder and remember ", style="bold #e2e8f0")
            body.append("(Enter)\n", style="#64748b")
            body.append("  [2] ", style="bold #f59e0b")
            body.append("Open in read-only Plan Mode for this session\n", style="#e2e8f0")
            body.append("  [3] ", style="bold #ef4444")
            body.append("No, cancel", style="#e2e8f0")
            self.console.print()
            self.console.print(
                Panel(
                    body,
                    title="[bold #f59e0b]🔒 Folder Trust Gate[/]",
                    title_align="left",
                    border_style="#f59e0b",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
        else:
            print(f"\n[TRUST FOLDER] {folder_path}\n  [1] Yes, trust (Enter)\n  [2] Read-only (/plan)\n  [3] Cancel")
        try:
            ans = input("  Choice [1/2/3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        if ans in ("", "1", "y", "yes", "e", "evet"):
            return "trust"
        if ans in ("2", "p", "plan"):
            return "plan"
        return "deny"

    def pick_project_interactive(
        self,
        projects: list[dict[str, Any]],
        current_path: str,
        *,
        title: str = "📁 Select Project Folder",
    ) -> str | None:
        """Interactive numbered project selector. Returns chosen path, 'new:<name>', or None (cancel/keep)."""
        if self.console:
            tbl = Table(box=box.ROUNDED, border_style="#334155", padding=(0, 1))
            tbl.add_column("#", style="bold #38bdf8", justify="right", no_wrap=True)
            tbl.add_column("Project", style="bold #f8fafc", no_wrap=True)
            tbl.add_column("Status", no_wrap=True)
            tbl.add_column("Path", style="#64748b")
            for idx, p in enumerate(projects, 1):
                is_cur = os.path.normcase(p["path"]) == os.path.normcase(current_path)
                tags = []
                if is_cur:
                    tags.append("[bold #22c55e]● Active[/]")
                elif p.get("trusted"):
                    tags.append("[#22c55e]🔒 Trusted[/]")
                else:
                    tags.append("[#f59e0b]○ New[/]")
                if p.get("is_git"):
                    tags.append("[#a78bfa]⎇ git[/]")
                tbl.add_row(str(idx), p["name"], " ".join(tags), p["path"])
            self.console.print()
            self.console.print(
                Panel(
                    tbl,
                    title=f"[bold #38bdf8]{title}[/]",
                    subtitle="[#64748b]Enter number / project name / path · type `new <name>` to create · Enter = current[/]",
                    border_style="#2563eb",
                    box=box.ROUNDED,
                )
            )
        else:
            print(f"\n{title}")
            for idx, p in enumerate(projects, 1):
                print(f"  {idx}. {p['name']} ({p['path']})")
        try:
            raw = input("  Select project (Enter=current): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return None
        if raw.lower().startswith("new "):
            return "new:" + raw[4:].strip()
        if raw.isdigit():
            i = int(raw) - 1
            if 0 <= i < len(projects):
                return projects[i]["path"]
        for p in projects:
            if p["name"].lower() == raw.lower():
                return p["path"]
        return raw

    def print_files_table(self, files: list[tuple[str, int]], pinned: list[str], subdir: str = "") -> None:
        if not self.console:
            for rel, sz in files[:60]:
                pin = " [PIN]" if rel in pinned else ""
                print(f"  {rel} ({sz} B){pin}")
            return
        tbl = Table(box=box.ROUNDED, border_style="#334155", padding=(0, 1))
        tbl.add_column("File", style="#e2e8f0")
        tbl.add_column("Size", justify="right", style="#64748b", no_wrap=True)
        tbl.add_column("Context", no_wrap=True)
        for rel, sz in files[:45]:
            sz_str = f"{sz / 1024:.1f} KB" if sz >= 1024 else f"{sz} B"
            pin = "[bold #38bdf8]📎 Pinned[/]" if rel in pinned else "[#475569]—[/]"
            tbl.add_row(rel, sz_str, pin)
        extra = f" · first 45/{len(files)} files" if len(files) > 45 else f" · {len(files)} files"
        sub_lbl = f" ({subdir})" if subdir else ""
        self.console.print(
            Panel(
                tbl,
                title=f"[bold #38bdf8]📂 Project Files{sub_lbl}[/]",
                subtitle=f"[#64748b]Pin to context: /add <file> (Tab) · unpin: /drop{extra}[/]",
                border_style="#334155",
                box=box.ROUNDED,
            )
        )

    def print_trusted_table(self, folders: list[str], current_path: str) -> None:
        if not self.console:
            print("Trusted folders:", folders)
            return
        tbl = Table(box=box.ROUNDED, border_style="#334155", show_header=False, padding=(0, 1))
        tbl.add_column("State", no_wrap=True)
        tbl.add_column("Path", style="#e2e8f0")
        if not folders:
            tbl.add_row("[#64748b]—[/]", "[#64748b]No trusted folders yet.[/]")
        for f in folders:
            cur = "[bold #22c55e]● Active[/]" if os.path.normcase(f) == os.path.normcase(current_path) else "[#38bdf8]🔒[/]"
            tbl.add_row(cur, f)
        self.console.print(
            Panel(
                tbl,
                title="[bold #22c55e]🔒 Trusted Folders (~/.picocode_trusted_folders.json)[/]",
                subtitle="[#64748b]Remove: /trust remove  ·  Add: /trust add[/]",
                border_style="#334155",
                box=box.ROUNDED,
            )
        )

    def print_tool_start(self, name: str, args: dict[str, Any]) -> None:
        label, color = TOOL_DISPLAY_NAMES.get(name, (name, "#38bdf8"))
        preview = format_tool_arg_preview(name, args)
        if self.console:
            t = Text()
            t.append("  ● ", style=f"bold {color}")
            t.append(label, style="bold #f8fafc")
            t.append(f"({preview})", style="#94a3b8")
            self.console.print(t)
        else:
            print(f"  ● {label}({preview})")

    def print_tool_result(self, summary: str, is_error: bool = False) -> None:
        if self.console:
            col = "#f87171" if is_error else "#64748b"
            t = Text()
            t.append("    ⎿  ", style="#334155")
            t.append(summary[:140], style=col)
            self.console.print(t)
        else:
            print(f"    ⎿  {summary[:140]}")

    def print_diff_card(self, path: str, added: int, removed: int, diff_txt: str, syntax_err: str = "") -> None:
        if not self.console:
            print(f"    ⎿  {path} (+{added}/-{removed})")
            if diff_txt:
                print(diff_txt[:800])
            return
        body = Text()
        lines = (diff_txt or "").splitlines()[:30]
        for idx, ln in enumerate(lines):
            if idx:
                body.append("\n")
            if ln.startswith("+") and not ln.startswith("+++"):
                body.append(ln, style="#86efac on #0c291a")
            elif ln.startswith("-") and not ln.startswith("---"):
                body.append(ln, style="#fca5a5 on #2d1216")
            elif ln.startswith("@@"):
                body.append(ln, style="bold #7dd3fc")
            elif ln.startswith("+++") or ln.startswith("---"):
                body.append(ln, style="#475569")
            else:
                body.append(ln, style="#94a3b8")
        if syntax_err:
            body.append(f"\n⚠ {syntax_err}", style="bold #f87171")
        title = f"[bold #34d399]✏ {rich_escape(path)}[/]  [bold #22c55e]+{added}[/] [bold #ef4444]-{removed}[/]"
        self.console.print(
            Panel(body, title=title, title_align="left", border_style="#1e3a5f", box=box.ROUNDED, padding=(0, 1))
        )

    def confirm_command(self, cmd: str, *, input_fn: Callable[[str], str] | None = None) -> str:
        """Ask user to approve shell command: returns 'y', 'a' (always this session), or 'n'."""
        if self.console:
            self.console.print(
                Panel(
                    Text(cmd, style="bold #fde047"),
                    title="[bold #f59e0b]⚡ Shell Command Approval[/]",
                    title_align="left",
                    border_style="#f59e0b",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            prompt_str = "  [y] Run  ·  [a] Always allow this session  ·  [n / Enter] Deny: "
        else:
            prompt_str = f"  Run `{cmd}`? [y/a/N]: "
        reader = input_fn or input
        try:
            ans = reader(prompt_str).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "n"
        if ans in ("h", "hayir", "hayır", "n", "no", ""):
            return "n"
        if ans in ("a", "always", "t", "tumu", "tümü"):
            return "a"
        if ans in ("y", "yes", "e", "evet"):
            return "y"
        return "n"

    def print_turn_footer(self, *, elapsed: float, chunks: int, tools: int, edits: int, ctx_tokens: int, ctx_limit: int) -> None:
        tps = chunks / max(0.05, elapsed) if chunks else 0.0
        pct = min(100, int(round((ctx_tokens / max(1, ctx_limit)) * 100)))
        k_str = f"{ctx_tokens / 1000:.1f}k" if ctx_tokens >= 1000 else str(ctx_tokens)
        parts = [f"{elapsed:.1f}s"]
        if chunks:
            parts.append(f"~{tps:.1f} tok/s")
        if tools:
            parts.append(f"{tools} tool{'s' if tools != 1 else ''}")
        parts.append(f"ctx {k_str}/{ctx_limit // 1000}k ({pct}%)")
        if edits:
            parts.append(f"{edits} edit{'s' if edits != 1 else ''} (/undo)")
        line = " · ".join(parts)
        if self.console:
            self.console.print(f"  [#475569]✓ {line}[/]\n")
        else:
            print(f"  ✓ {line}\n")

    def make_live_stream(self) -> Any:
        """Return a Rich Live context manager for animated spinner + live Markdown streaming."""
        if not self.console:
            return None
        return Live(
            Spinner("dots", text=Text(" Thinking...", style="#38bdf8"), style="#38bdf8"),
            console=self.console,
            refresh_per_second=14,
            transient=False,
        )

    def render_spinner(self, text: str) -> Any:
        if not self.console:
            return text
        return Spinner("dots", text=Text(f" {text}", style="#38bdf8"), style="#38bdf8")

    def render_markdown(self, md_text: str) -> Any:
        if not self.console:
            return md_text
        return Markdown(md_text, code_theme="monokai")
