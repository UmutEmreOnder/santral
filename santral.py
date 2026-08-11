#!/usr/bin/env python3
"""Santral: point any OpenAI-compatible app at your local CLI agents.

Exposes /v1/chat/completions on 127.0.0.1; the request's "model" field
selects which CLI agent (claude, codex, gemini, ...) runs the prompt.
"""

import json
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULTS = {
    "port": 8765,
    "sync_wait": 25,
    "agent_timeout": 600,
    "default_agent": "claude",
    "session_window": 300,
    "prompt_prefix": "",
}

BUILTIN_AGENTS = {
    "claude": {
        "command": ["claude", "-p", "--session-id", "{session_id}"],
        "resume_command": ["claude", "-p", "--resume", "{session_id}"],
        "model_args": ["--model", "{model}"],
        "effort_args": ["--effort", "{effort}"],
    },
    "codex": {
        # "-" reads the prompt from stdin; resume targets codex's most
        # recent recorded session, so no session id is needed.
        "command": ["codex", "exec", "-"],
        "resume_command": ["codex", "exec", "resume", "--last", "-"],
        "model_args": ["-m", "{model}"],
        "effort_args": ["-c", "model_reasoning_effort={effort}"],
    },
    "gemini": {
        "command": ["gemini", "-p", "{prompt}"],
        "resume_command": ["gemini", "--resume", "latest", "-p", "{prompt}"],
        "model_args": ["-m", "{model}"],
    },
}

CONFIG_PATH = Path.home() / ".config" / "santral" / "config.toml"


def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULTS)
    agents = {name: dict(agent) for name, agent in BUILTIN_AGENTS.items()}
    user = {}
    if path is not None and Path(path).is_file():
        user = tomllib.loads(Path(path).read_text())
    for key in DEFAULTS:
        if key in user:
            cfg[key] = user[key]
    for name, fields in user.get("agents", {}).items():
        agents.setdefault(name, {}).update(fields)
    for name, agent in agents.items():
        if "command" not in agent:
            raise ValueError(f"agent {name!r} has no 'command'")
        for value_key, args_key in (("model", "model_args"), ("effort", "effort_args")):
            if value_key in agent and args_key not in agent:
                print(f"warning: agent {name!r} sets '{value_key}' but has no "
                      f"'{args_key}' template; value ignored", file=sys.stderr)
                del agent[value_key]
    cfg["agents"] = agents
    return cfg


def build_argv(agent, resume, prompt, session_id):
    """Build the subprocess argv; returns (argv, stdin_input)."""
    argv = list(agent["resume_command"] if resume else agent["command"])
    for value_key, args_key in (("model", "model_args"), ("effort", "effort_args")):
        if value_key in agent:
            argv += [arg.replace("{" + value_key + "}", agent[value_key])
                     for arg in agent[args_key]]
    argv = [arg.replace("{session_id}", session_id) for arg in argv]
    stdin_input = prompt
    if any("{prompt}" in arg for arg in argv):
        argv = [arg.replace("{prompt}", prompt) for arg in argv]
        stdin_input = None
    return argv, stdin_input


class SessionTracker:
    """Per-agent last-session memory for resume-within-window decisions."""

    def __init__(self, window):
        self.window = window
        self._lock = threading.Lock()
        self._sessions = {}  # agent name -> {"id": str, "finished": float | None}

    def begin(self, name, has_resume):
        with self._lock:
            prev = self._sessions.get(name)
            if (has_resume and self.window > 0 and prev
                    and prev["finished"] is not None
                    and time.monotonic() - prev["finished"] < self.window):
                prev["finished"] = None  # session is busy again
                return True, prev["id"]
            session_id = str(uuid.uuid4())
            self._sessions[name] = {"id": session_id, "finished": None}
            return False, session_id

    def finish(self, name, session_id):
        with self._lock:
            prev = self._sessions.get(name)
            if prev and prev["id"] == session_id:
                prev["finished"] = time.monotonic()


def notify(title, body):
    try:
        subprocess.run(["notify-send", "-a", "Santral", title, body[:1000]],
                       check=False)
    except FileNotFoundError:
        print(f"[notify] {title}: {body}")


def copy_to_clipboard(text):
    for argv in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
        try:
            # wl-copy/xclip fork a child that holds the clipboard and keeps
            # inherited pipes open; capturing output would block until the
            # clipboard is replaced, so send it to DEVNULL instead.
            proc = subprocess.run(argv, input=text, text=True,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  timeout=10)
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            continue
    return False


def run_agent(cfg, name, argv, stdin_input, tracker, session_id, done, result):
    try:
        proc = subprocess.run(argv, input=stdin_input, capture_output=True,
                              text=True, timeout=cfg["agent_timeout"])
        out = proc.stdout.strip() or proc.stderr.strip() or "(empty response)"
    except subprocess.TimeoutExpired:
        out = f"{name} timed out after {cfg['agent_timeout']} seconds."
    except FileNotFoundError:
        out = f"{argv[0]} CLI not found (is it on PATH?)."
    except Exception as exc:
        out = f"{name} failed: {exc}"
    tracker.finish(name, session_id)
    result["text"] = out
    done.set()
    if result.get("backgrounded"):
        if copy_to_clipboard(out):
            notify("Agent finished",
                   f"{name} finished its job, you can paste the result.")
        else:
            notify("Agent finished", out)


def make_handler(cfg, tracker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse(self, name, text):
            def chunk(delta, finish):
                return json.dumps({
                    "id": "santral",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": name,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}],
                })
            body = (f"data: {chunk({'role': 'assistant', 'content': text}, None)}\n\n"
                    f"data: {chunk({}, 'stop')}\n\n"
                    "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # OpenAI (/v1/models) and Ollama (/api/tags) model-listing probes
            if self.path in ("/v1/models", "/api/tags", "/"):
                names = sorted(cfg["agents"])
                self._json(200, {
                    "object": "list",
                    "data": [{"id": n, "object": "model"} for n in names],
                    "models": [{"name": n} for n in names],
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "bad json"})
                return

            if not isinstance(req, dict):
                self._json(400, {"error": "bad json"})
                return

            user_text = ""
            for msg in req.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "user":
                    user_text = msg.get("content") or user_text
            if not user_text:
                self._json(400, {"error": "no user message"})
                return

            model = req.get("model") or ""
            name = model if model in cfg["agents"] else cfg["default_agent"]
            agent = cfg["agents"][name]

            prompt = user_text
            if cfg["prompt_prefix"]:
                prompt = cfg["prompt_prefix"] + "\n\n" + user_text

            resume, session_id = tracker.begin(name, "resume_command" in agent)
            argv, stdin_input = build_argv(agent, resume, prompt, session_id)

            done = threading.Event()
            result = {}
            threading.Thread(
                target=run_agent,
                args=(cfg, name, argv, stdin_input, tracker, session_id,
                      done, result),
                daemon=True).start()
            if done.wait(cfg["sync_wait"]):
                text = result["text"]
            else:
                result["backgrounded"] = True
                text = (f"{name} is running in the background; "
                        "the result will be copied to your clipboard when ready.")

            if req.get("stream"):
                self._sse(name, text)
                return
            self._json(200, {
                "id": "santral",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            })

    return Handler


def main():
    cfg = load_config()
    tracker = SessionTracker(cfg["session_window"])
    server = ThreadingHTTPServer(("127.0.0.1", cfg["port"]),
                                 make_handler(cfg, tracker))
    print(f"santral listening on 127.0.0.1:{cfg['port']} "
          f"(agents: {', '.join(sorted(cfg['agents']))})")
    server.serve_forever()


if __name__ == "__main__":
    main()
