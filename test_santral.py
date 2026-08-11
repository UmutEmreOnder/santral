import contextlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import santral


class TestLoadConfig(unittest.TestCase):
    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return Path(f.name)

    def test_defaults_without_config_file(self):
        cfg = santral.load_config(Path("/nonexistent/config.toml"))
        self.assertEqual(cfg["port"], 8765)
        self.assertEqual(cfg["sync_wait"], 25)
        self.assertEqual(cfg["agent_timeout"], 600)
        self.assertEqual(cfg["default_agent"], "claude")
        self.assertEqual(cfg["session_window"], 300)
        self.assertEqual(set(cfg["agents"]), {"claude", "codex", "gemini"})
        self.assertEqual(cfg["agents"]["claude"]["command"][:2], ["claude", "-p"])

    def test_top_level_override(self):
        path = self._write('port = 9999\nsession_window = 60\n')
        cfg = santral.load_config(path)
        self.assertEqual(cfg["port"], 9999)
        self.assertEqual(cfg["session_window"], 60)
        self.assertEqual(cfg["sync_wait"], 25)  # untouched default

    def test_agent_per_field_merge(self):
        path = self._write('[agents.claude]\nmodel = "opus"\n')
        cfg = santral.load_config(path)
        agent = cfg["agents"]["claude"]
        self.assertEqual(agent["model"], "opus")
        # built-in command survives a partial override
        self.assertEqual(agent["command"][:2], ["claude", "-p"])

    def test_new_custom_agent(self):
        path = self._write('[agents.myagent]\ncommand = ["my-agent", "{prompt}"]\n')
        cfg = santral.load_config(path)
        self.assertEqual(cfg["agents"]["myagent"]["command"], ["my-agent", "{prompt}"])

    def test_new_agent_without_command_rejected(self):
        path = self._write('[agents.broken]\nmodel = "x"\n')
        with self.assertRaises(ValueError):
            santral.load_config(path)

    def test_value_without_args_template_warns_and_ignores(self):
        # gemini has no effort_args built in
        path = self._write('[agents.gemini]\neffort = "high"\n')
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cfg = santral.load_config(path)
        self.assertNotIn("effort", cfg["agents"]["gemini"])
        self.assertIn("effort_args", stderr.getvalue())


class TestBuildArgv(unittest.TestCase):
    def test_prompt_via_stdin_when_no_placeholder(self):
        agent = {"command": ["claude", "-p", "--session-id", "{session_id}"]}
        argv, stdin_input = santral.build_argv(agent, False, "hello", "SID")
        self.assertEqual(argv, ["claude", "-p", "--session-id", "SID"])
        self.assertEqual(stdin_input, "hello")

    def test_prompt_substituted_into_arg(self):
        agent = {"command": ["gemini", "-p", "{prompt}"]}
        argv, stdin_input = santral.build_argv(agent, False, "hello", "SID")
        self.assertEqual(argv, ["gemini", "-p", "hello"])
        self.assertIsNone(stdin_input)

    def test_resume_uses_resume_command(self):
        agent = {"command": ["a", "{session_id}"],
                 "resume_command": ["a", "--resume", "{session_id}"]}
        argv, _ = santral.build_argv(agent, True, "x", "SID")
        self.assertEqual(argv, ["a", "--resume", "SID"])

    def test_model_and_effort_fragments_appended(self):
        agent = {"command": ["codex", "exec", "-"],
                 "resume_command": ["codex", "exec", "resume", "--last", "-"],
                 "model_args": ["-m", "{model}"],
                 "effort_args": ["-c", "model_reasoning_effort={effort}"],
                 "model": "gpt-5", "effort": "high"}
        argv, _ = santral.build_argv(agent, False, "x", "SID")
        self.assertEqual(argv, ["codex", "exec", "-", "-m", "gpt-5",
                                "-c", "model_reasoning_effort=high"])
        argv_resume, _ = santral.build_argv(agent, True, "x", "SID")
        self.assertEqual(argv_resume[:5], ["codex", "exec", "resume", "--last", "-"])
        self.assertIn("-m", argv_resume)

    def test_fragments_not_appended_without_value(self):
        agent = {"command": ["gemini", "-p", "{prompt}"], "model_args": ["-m", "{model}"]}
        argv, _ = santral.build_argv(agent, False, "x", "SID")
        self.assertEqual(argv, ["gemini", "-p", "x"])


import time


class TestSessionTracker(unittest.TestCase):
    def test_first_run_is_fresh(self):
        t = santral.SessionTracker(window=300)
        resume, sid = t.begin("a", has_resume=True)
        self.assertFalse(resume)
        self.assertTrue(sid)

    def test_resume_within_window_reuses_id(self):
        t = santral.SessionTracker(window=300)
        _, sid = t.begin("a", has_resume=True)
        t.finish("a", sid)
        resume, sid2 = t.begin("a", has_resume=True)
        self.assertTrue(resume)
        self.assertEqual(sid2, sid)

    def test_fresh_after_window_expires(self):
        t = santral.SessionTracker(window=0.05)
        _, sid = t.begin("a", has_resume=True)
        t.finish("a", sid)
        time.sleep(0.1)
        resume, sid2 = t.begin("a", has_resume=True)
        self.assertFalse(resume)
        self.assertNotEqual(sid2, sid)

    def test_busy_agent_gets_fresh_session(self):
        t = santral.SessionTracker(window=300)
        _, sid = t.begin("a", has_resume=True)
        # previous run has not finished
        resume, sid2 = t.begin("a", has_resume=True)
        self.assertFalse(resume)
        self.assertNotEqual(sid2, sid)
        # stale finish of the replaced run must not corrupt state
        t.finish("a", sid)
        t.finish("a", sid2)
        resume3, sid3 = t.begin("a", has_resume=True)
        self.assertTrue(resume3)
        self.assertEqual(sid3, sid2)

    def test_no_resume_command_never_resumes(self):
        t = santral.SessionTracker(window=300)
        _, sid = t.begin("a", has_resume=False)
        t.finish("a", sid)
        resume, _ = t.begin("a", has_resume=False)
        self.assertFalse(resume)

    def test_window_zero_disables_resume(self):
        t = santral.SessionTracker(window=0)
        _, sid = t.begin("a", has_resume=True)
        t.finish("a", sid)
        resume, _ = t.begin("a", has_resume=True)
        self.assertFalse(resume)

    def test_agents_tracked_independently(self):
        t = santral.SessionTracker(window=300)
        _, sid_a = t.begin("a", has_resume=True)
        t.finish("a", sid_a)
        resume_b, _ = t.begin("b", has_resume=True)
        self.assertFalse(resume_b)


def _run(cfg, name, argv, stdin_input, tracker=None, session_id="SID",
         backgrounded=False):
    tracker = tracker or santral.SessionTracker(window=300)
    done = threading.Event()
    result = {"backgrounded": True} if backgrounded else {}
    santral.run_agent(cfg, name, argv, stdin_input, tracker, session_id, done, result)
    return done, result


class TestRunAgent(unittest.TestCase):
    CFG = {"agent_timeout": 5}

    def test_captures_stdout(self):
        done, result = _run(self.CFG, "fake", ["cat"], "hello world")
        self.assertTrue(done.is_set())
        self.assertEqual(result["text"], "hello world")

    def test_missing_binary_reports_error(self):
        done, result = _run(self.CFG, "fake", ["definitely-not-a-real-cli-xyz"], "x")
        self.assertTrue(done.is_set())
        self.assertIn("not found", result["text"])

    def test_empty_output_placeholder(self):
        done, result = _run(self.CFG, "fake", ["true"], None)
        self.assertEqual(result["text"], "(empty response)")

    def test_timeout_reports_error(self):
        cfg = {"agent_timeout": 0.1}
        done, result = _run(cfg, "fake", ["sleep", "5"], None)
        self.assertIn("timed out", result["text"])

    def test_backgrounded_run_notifies(self):
        calls = []
        orig = santral.notify
        santral.notify = lambda title, body: calls.append((title, body))
        self.addCleanup(setattr, santral, "notify", orig)
        _run(self.CFG, "fake", ["cat"], "late answer", backgrounded=True)
        self.assertEqual(calls, [("fake finished", "late answer")])

    def test_marks_session_finished(self):
        tracker = santral.SessionTracker(window=300)
        _, sid = tracker.begin("fake", has_resume=True)
        _run(self.CFG, "fake", ["cat"], "x", tracker=tracker, session_id=sid)
        resume, sid2 = tracker.begin("fake", has_resume=True)
        self.assertTrue(resume)
        self.assertEqual(sid2, sid)

    def test_permission_error_caught(self):
        # Create a non-executable file to trigger PermissionError
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(b"not executable")
        f.close()
        self.addCleanup(Path(f.name).unlink)
        done, result = _run(self.CFG, "fake", [f.name], None)
        self.assertTrue(done.is_set())
        self.assertIn("failed", result["text"])


def _start_server(test, cfg):
    """Start santral's handler on an ephemeral port; returns the port."""
    tracker = santral.SessionTracker(cfg["session_window"])
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 santral.make_handler(cfg, tracker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return server.server_address[1]


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as r:
        return r.status, json.loads(r.read())


def _base_cfg(**overrides):
    cfg = santral.load_config(Path("/nonexistent/config.toml"))
    cfg.update(overrides)
    return cfg


class TestGetEndpoints(unittest.TestCase):
    def setUp(self):
        self.port = _start_server(self, _base_cfg())

    def test_v1_models_lists_agents(self):
        status, body = _get(self.port, "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(m["id"] for m in body["data"]),
                         ["claude", "codex", "gemini"])
        self.assertEqual(sorted(m["name"] for m in body["models"]),
                         ["claude", "codex", "gemini"])

    def test_api_tags_and_root_also_list(self):
        for path in ("/api/tags", "/"):
            status, body = _get(self.port, path)
            self.assertEqual(status, 200)
            self.assertIn("models", body)

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _get(self.port, "/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_v1_models_with_suffix_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _get(self.port, "/v1/models/foo")
        self.assertEqual(ctx.exception.code, 404)


def _post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


def _chat(model, text):
    return {"model": model, "messages": [{"role": "user", "content": text}]}


class TestChatCompletions(unittest.TestCase):
    def setUp(self):
        # notify would pop a real desktop notification during async tests
        orig = santral.notify
        santral.notify = lambda title, body: None
        self.addCleanup(setattr, santral, "notify", orig)
        cfg = _base_cfg(sync_wait=2, default_agent="echo")
        cfg["agents"]["echo"] = {"command": ["cat"]}
        self.port = _start_server(self, cfg)

    def test_sync_answer(self):
        status, body = _post(self.port, _chat("echo", "hello santral"))
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "hello santral")
        self.assertEqual(body["model"], "echo")
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_uses_latest_user_message(self):
        payload = {"model": "echo", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]}
        _, body = _post(self.port, payload)
        self.assertEqual(body["choices"][0]["message"]["content"], "second")

    def test_unknown_model_routes_to_default(self):
        _, body = _post(self.port, _chat("gpt-4o", "fallback test"))
        self.assertEqual(body["model"], "echo")
        self.assertEqual(body["choices"][0]["message"]["content"], "fallback test")

    def test_async_ack_when_agent_is_slow(self):
        cfg = _base_cfg(sync_wait=0.2, default_agent="slow")
        cfg["agents"]["slow"] = {"command": ["sh", "-c", "sleep 2; echo done"]}
        port = _start_server(self, cfg)
        _, body = _post(port, _chat("slow", "x"))
        self.assertIn("running in the background", body["choices"][0]["message"]["content"])

    def test_bad_json_400(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=b"{not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_no_user_message_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _post(self.port, {"model": "echo", "messages": []})
        self.assertEqual(ctx.exception.code, 400)

    def test_non_object_body_400(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps([1, 2, 3]).encode(),
            headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_non_dict_message_entry_skipped(self):
        payload = {
            "model": "echo",
            "messages": [
                "not a dict",
                {"role": "user", "content": "valid message"}
            ]
        }
        status, body = _post(self.port, payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "valid message")


class TestStreaming(unittest.TestCase):
    def setUp(self):
        cfg = _base_cfg(sync_wait=2, default_agent="echo")
        cfg["agents"]["echo"] = {"command": ["cat"]}
        self.port = _start_server(self, cfg)

    def test_single_chunk_sse(self):
        payload = _chat("echo", "stream me")
        payload["stream"] = True
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.headers["Content-Type"], "text/event-stream")
            raw = r.read().decode()
        events = [line[len("data: "):] for line in raw.split("\n")
                  if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        first = json.loads(events[0])
        self.assertEqual(first["object"], "chat.completion.chunk")
        self.assertEqual(first["choices"][0]["delta"]["content"], "stream me")
        self.assertIsNone(first["choices"][0]["finish_reason"])
        second = json.loads(events[1])
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")
        self.assertEqual(second["choices"][0]["delta"], {})


class TestSessionContinuityHTTP(unittest.TestCase):
    def _server_with_recorder(self, window):
        self.log = Path(tempfile.mkdtemp()) / "calls.log"
        cfg = _base_cfg(sync_wait=5, session_window=window, default_agent="rec")
        cfg["agents"]["rec"] = {
            "command": ["sh", "-c",
                        f"echo fresh {{session_id}} >> {self.log}; echo ok"],
            "resume_command": ["sh", "-c",
                               f"echo resume {{session_id}} >> {self.log}; echo ok"],
        }
        return _start_server(self, cfg)

    def _lines(self):
        return self.log.read_text().splitlines()

    def test_second_request_resumes_with_same_id(self):
        port = self._server_with_recorder(window=300)
        _post(port, _chat("rec", "one"))
        _post(port, _chat("rec", "two"))
        lines = self._lines()
        self.assertEqual(len(lines), 2)
        mode1, sid1 = lines[0].split()
        mode2, sid2 = lines[1].split()
        self.assertEqual((mode1, mode2), ("fresh", "resume"))
        self.assertEqual(sid1, sid2)

    def test_fresh_after_window_expiry(self):
        port = self._server_with_recorder(window=0.05)
        _post(port, _chat("rec", "one"))
        time.sleep(0.2)
        _post(port, _chat("rec", "two"))
        lines = self._lines()
        mode1, sid1 = lines[0].split()
        mode2, sid2 = lines[1].split()
        self.assertEqual((mode1, mode2), ("fresh", "fresh"))
        self.assertNotEqual(sid1, sid2)

    def test_window_zero_never_resumes(self):
        port = self._server_with_recorder(window=0)
        _post(port, _chat("rec", "one"))
        _post(port, _chat("rec", "two"))
        modes = [line.split()[0] for line in self._lines()]
        self.assertEqual(modes, ["fresh", "fresh"])


if __name__ == "__main__":
    unittest.main()
