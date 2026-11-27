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
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

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
    ("/model", "Kurulu modelleri gör veya canlı değiştir (/model inkling2-mini)"),
    ("/plan", "Salt-okunur Plan Modu aç / kapat"),
    ("/diff", "Çalışma dizinindeki Git diff özetini göster"),
    ("/undo", "Son turdaki tüm dosya değişikliklerini geri al"),
    ("/status", "Model, bağlam penceresi, Git ve durum özeti"),
    ("/yes", "Kabuk komutları için otomatik onayı aç / kapat"),
    ("/clear", "Sohbet geçmişini sıfırla ve ekranı temizle"),
    ("/help", "Komut listesini ve kısayolları göster"),
    ("/exit", "PicoCode Terminal Ajanından çık"),
]

TOOL_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
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
        """Interactive dropdown autocomplete for `/` slash commands, `/model <id>`, and `@file` references."""

        def __init__(
            self,
            get_files_fn: Callable[[], list[str]],
            get_models_fn: Callable[[], list[dict[str, Any]]] | None = None,
        ) -> None:
            self._get_files = get_files_fn
            self._get_models = get_models_fn or (lambda: [])

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            # 1) `/model <id>` sub-completion
            if text.lower().startswith("/model "):
                partial = text[7:].strip().lower()
                for m in self._get_models():
                    mid = str(m.get("id") or "")
                    if partial in mid.lower():
                        tag = "✓ kurulu" if m.get("installed") else "eksik"
                        yield Completion(
                            mid,
                            start_position=-len(text[7:]),
                            display=mid,
                            display_meta=f"{tag} · {m.get('short', '')}",
                        )
                return
            # 2) Slash commands at start of line
            if text.startswith("/") and " " not in text:
                q = text.lower()
                for cmd, desc in SLASH_COMMANDS:
                    if cmd.startswith(q):
                        yield Completion(cmd, start_position=-len(text), display=cmd, display_meta=desc)
                return
            # 3) @file reference anywhere in prompt
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
                            display_meta="dosya bağlamı",
                        )


class TerminalUI:
    """Rich + prompt_toolkit TUI facade with clean stdlib fallback."""

    def __init__(
        self,
        get_files_fn: Callable[[], list[str]] | None = None,
        get_models_fn: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        init_windows_console()
        self.rich_enabled = HAS_RICH
        self.console = Console(highlight=False) if HAS_RICH else None
        self._get_files = get_files_fn or (lambda: [])
        self._get_models = get_models_fn or (lambda: [])
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
                completer=PicoCompleter(self._get_files, self._get_models),
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

        mode_str = "[bold #f59e0b]◐ PLAN (salt-okunur)[/]" if plan_only else "[bold #22c55e]● AGENT (16 adım)[/]"
        if auto_approve:
            mode_str += "  [bold #38bdf8]⚡ --yes[/]"
        git_str = f"[#a78bfa]⎇ {git_branch}[/]" if git_branch else "[#64748b]git yok[/]"
        if git_branch:
            git_str += f"  [#f59e0b]({git_changes} değişiklik)[/]" if git_changes else "  [#22c55e](temiz)[/]"

        meta.add_row("Model", f"[bold #38bdf8]{model}[/]  [#64748b](32K ctx · ngram-mod)[/]")
        meta.add_row("Proje", f"[bold #f8fafc]{workspace_name}[/]  [#64748b]{workspace_path}[/]")
        meta.add_row("Git", git_str)
        meta.add_row("Mod", mode_str)
        meta.add_row(
            "Kısayol",
            "[#38bdf8]/model[/]  [#38bdf8]/plan[/]  [#38bdf8]/diff[/]  [#38bdf8]/undo[/]  [#38bdf8]@dosya[/]  [#64748b]· Tab tamamla · Ctrl+C iptal[/]",
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
            subtitle="[#64748b]Apache-2.0 · 100% Yerel[/]",
            subtitle_align="right",
        )
        self.console.print()
        self.console.print(panel)

    def read_prompt(
        self,
        *,
        model: str,
        git_branch: str,
        ctx_tokens: int,
        ctx_limit: int,
        plan_only: bool,
        auto_approve: bool,
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
            branch_part = f"  <sep>│</sep>  ⎇ {git_branch}" if git_branch else ""
            ctx_cls = "bottom-toolbar.warn" if pct >= 65 else "bottom-toolbar.key"

            def _toolbar():
                return HTML(
                    f" <{mode_cls}>● {mode_lbl}{yes_lbl}</{mode_cls}>"
                    f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  <bottom-toolbar.key>{model}</bottom-toolbar.key>"
                    f"{branch_part.replace('<sep>', '<bottom-toolbar.sep>').replace('</sep>', '</bottom-toolbar.sep>')}"
                    f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  ctx <{ctx_cls}>{k_str} ({pct}%)</{ctx_cls}>"
                    f"  <bottom-toolbar.sep>│</bottom-toolbar.sep>  /help · @dosya · Tab tamamla"
                )

            rprompt = HTML("<rprompt>Ctrl+C çıkış</rprompt>")
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
            self.console.print(f"  [bold #22c55e]✓[/] [#e2e8f0]{msg}[/]")
        else:
            print(f"  ✓ {msg}")

    def print_warn(self, msg: str) -> None:
        if self.console:
            self.console.print(f"  [bold #f59e0b]▲[/] [#fde68a]{msg}[/]")
        else:
            print(f"  ▲ {msg}")

    def print_error(self, msg: str) -> None:
        if self.console:
            self.console.print(f"  [bold #ef4444]✗[/] [#fca5a5]{msg}[/]")
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
        tbl.add_row("@dosya", "Prompt içine doğrudan dosya içeriği ekle (Tab ile seç)")
        self.console.print(Panel(tbl, title="[bold #38bdf8]Komutlar[/]", border_style="#334155", box=box.ROUNDED))

    def print_models_table(self, models: list[dict[str, Any]], active_model: str) -> None:
        if not self.console:
            for m in models:
                st = "*" if m["id"] == active_model else ("OK" if m["installed"] else "MISSING")
                print(f"  [{st}] {m['id']} ({m['ctx_limit'] // 1024}K) - {m['label']}")
            return
        tbl = Table(box=box.ROUNDED, border_style="#334155", padding=(0, 1))
        tbl.add_column("Model ID", style="bold #38bdf8", no_wrap=True)
        tbl.add_column("Durum", no_wrap=True)
        tbl.add_column("Bağlam", justify="right", style="#94a3b8", no_wrap=True)
        tbl.add_column("Özellik", style="#cbd5e1")
        for m in models:
            if m["id"] == active_model:
                st = "[bold #22c55e]● Aktif[/]"
            elif m["installed"]:
                st = "[#38bdf8]✓ Kurulu[/]"
            else:
                st = "[#64748b]✗ Eksik[/]"
            tbl.add_row(m["id"], st, f"{m['ctx_limit'] // 1024}K", m.get("short") or m.get("label", ""))
        self.console.print(
            Panel(
                tbl,
                title="[bold #38bdf8]✦ Yerel Kodlama Modelleri[/]",
                subtitle="[#64748b]Canlı geçiş: /model inkling2-mini  veya  /model qwen3-coder-30b[/]",
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
        title = f"[bold #34d399]✏ {path}[/]  [bold #22c55e]+{added}[/] [bold #ef4444]-{removed}[/]"
        self.console.print(
            Panel(body, title=title, title_align="left", border_style="#1e3a5f", box=box.ROUNDED, padding=(0, 1))
        )

    def confirm_command(self, cmd: str) -> str:
        """Ask user to approve shell command: returns 'y', 'a' (always this session), or 'n'."""
        if self.console:
            self.console.print(
                Panel(
                    Text(cmd, style="bold #fde047"),
                    title="[bold #f59e0b]⚡ Kabuk Komutu Onayı[/]",
                    title_align="left",
                    border_style="#f59e0b",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            prompt_str = "  [e] Çalıştır  ·  [h] Bu oturumda hep izin ver  ·  [Enter/h-hayır] İptal: "
        else:
            prompt_str = f"  Run `{cmd}`? [y/a/N]: "
        try:
            ans = input(prompt_str).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "n"
        if ans in ("a", "always", "h", "hep"):
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
            parts.append(f"{tools} araç")
        parts.append(f"ctx {k_str}/{ctx_limit // 1000}k ({pct}%)")
        if edits:
            parts.append(f"{edits} düzenleme (/undo)")
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
            Spinner("dots", text=Text(" Düşünüyor...", style="#38bdf8"), style="#38bdf8"),
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
