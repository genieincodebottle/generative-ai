"""Focused service tests. All HTTP responses here are synthetic, not captures.

Run: python -m unittest discover -s tests -v
"""
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import requests
from services import search


def reply(result=None, request_id=1, status=200, sse=False):
    response = requests.Response()
    response.status_code = status
    message = {"jsonrpc": "2.0", "id": request_id, "result": result}
    text = json.dumps(message, ensure_ascii=False)
    response._content = ((f"event: message\r\ndata: {text}\r\n\r\n")
                         if sse else text).encode()
    response.headers["Content-Type"] = "text/event-stream" if sse else "application/json"
    return response


def hit(n=1):
    return {"url": f"https://example.com/{n}", "title": None,
            "excerpts": ["A passage\u2028with a Unicode separator.", "More evidence."]}


def exchange(payload=None):
    init = reply({"protocolVersion": "2025-03-26"}, sse=True)
    init.headers["Mcp-Session-Id"] = "synthetic-session"
    return [init, reply(status=202),
            reply({"tools": [{"name": "web_search"}]}, 2),
            reply(payload if payload is not None else {
                "structuredContent": {"results": [hit(1), hit(2), hit(3)]}}, 3, sse=True),
            reply(status=200)]


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_default_and_explicit_routing(self):
        self.assertEqual(search.default_provider(), "duckduckgo")
        self.assertEqual(search.available_providers(), ["duckduckgo", "parallel"])
        with patch.object(search, "_search_duckduckgo", return_value="ddg") as ddg, \
                patch.object(search, "_search_parallel", return_value="parallel") as parallel:
            self.assertEqual(search.search_sync("question"), "ddg")
            parallel.assert_not_called()
            self.assertEqual(search.search_sync("question", provider="parallel"), "parallel")
            ddg.assert_called_once()
        with patch.dict(os.environ, {"TAVILY_API_KEY": "explicit", "SERPAPI_API_KEY": "explicit"}):
            self.assertEqual(search.default_provider(), "tavily")
            with patch.object(search, "_search_serpapi", return_value="serp") as serp:
                self.assertEqual(search.search_sync("q", provider="serpapi", location="Paris"), "serp")
                serp.assert_called_once_with("q", 5, "Paris")
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "explicit"}):
            self.assertEqual(search.default_provider(), "serpapi")

    def test_missing_keys_do_not_fall_back(self):
        with patch.object(search, "_search_parallel") as parallel:
            for provider, key in [("tavily", "TAVILY_API_KEY"), ("serpapi", "SERPAPI_API_KEY")]:
                with self.assertRaisesRegex(search.SearchError, key):
                    search.search_sync("q", provider=provider)
            parallel.assert_not_called()

    def test_bad_incumbent_keys_do_not_fall_back(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "bad", "SERPAPI_API_KEY": "bad"}), \
                patch("requests.post", return_value=reply(status=401)), \
                patch("requests.get", return_value=reply(status=401)), \
                patch.object(search, "_search_parallel") as parallel:
            for provider in ("tavily", "serpapi"):
                with self.assertRaisesRegex(search.SearchError, "401"):
                    search.search_sync("q", provider=provider)
            parallel.assert_not_called()

    def test_validation_before_network(self):
        with patch("requests.Session.send") as send:
            for query, limit in [(" ", 1), ("q", 0), ("q", 26)]:
                with self.assertRaises(search.SearchError):
                    search.search_sync(query, limit, "parallel")
            with self.assertRaisesRegex(search.SearchError, "Unknown provider"):
                search.search_sync("q", provider="offline")
            send.assert_not_called()

    def test_protocol_headers_and_limit(self):
        sent = []
        responses = iter(exchange())

        def send(session, request, **kwargs):
            sent.append(request)
            self.assertFalse(kwargs.get("allow_redirects", False))
            return next(responses)

        with patch("requests.Session.send", new=send):
            result = search.search_sync("  question  ", 2, "parallel")
        self.assertEqual(result.query, "question")
        self.assertEqual(result.provider, "parallel")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0].title, "https://example.com/1")
        self.assertIn("\u2028", result.results[0].snippet)
        self.assertEqual([r.method for r in sent], ["POST"] * 4 + ["DELETE"])
        self.assertTrue(all(r.headers["User-Agent"] == search.PARALLEL_USER_AGENT for r in sent))
        self.assertTrue(all("Authorization" not in r.headers for r in sent))
        self.assertTrue(all(r.headers["Mcp-Session-Id"] == "synthetic-session" for r in sent[1:]))
        bodies = [json.loads(r.body) for r in sent[:4]]
        self.assertEqual([b["method"] for b in bodies],
                         ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        self.assertNotIn("id", bodies[1])
        self.assertEqual(bodies[3]["params"]["arguments"],
                         {"objective": "question", "search_queries": ["question"]})

    def test_explicit_parallel_key_and_http_errors(self):
        for status in (401, 403, 429, 500):
            responses = exchange()
            responses[3] = reply(status=status)
            with self.subTest(status=status), patch.dict(os.environ, {"PARALLEL_API_KEY": "synthetic-bad-key"}), \
                    patch("requests.Session.send", side_effect=responses) as send:
                with self.assertRaisesRegex(search.SearchError, str(status)):
                    search.search_sync("q", provider="parallel")
                self.assertEqual(send.call_count, 5)  # includes teardown, no retry
                for call in send.call_args_list:
                    self.assertEqual(call.args[0].headers["Authorization"], "Bearer synthetic-bad-key")

    def test_tool_error_and_cleanup(self):
        with patch("requests.Session.send", side_effect=exchange({
            "isError": True, "content": [{"type": "text", "text": "quota exhausted"}]})) as send:
            with self.assertRaisesRegex(search.SearchError, "quota exhausted"):
                search.search_sync("q", provider="parallel")
            self.assertEqual(send.call_count, 5)

    def test_text_payload_empty_and_malformed_results(self):
        result = search._parallel_results({"content": [{"type": "text", "text": json.dumps({"results": [hit()]})}]}, "q", 1)
        self.assertIn("More evidence", result.results[0].snippet)
        self.assertEqual(search._parallel_results({"structuredContent": {"results": []}}, "q", 1).results, [])
        for payload in [{}, {"structuredContent": {"results": None}},
                        {"structuredContent": {"results": [{"url": "javascript:bad", "excerpts": []}]}}]:
            with self.assertRaises(search.SearchError):
                search._parallel_results(payload, "q", 1)

    def test_sse_cr_lf_multiline_and_matching_id(self):
        for boundary in ("\n", "\r", "\r\n"):
            response = reply(sse=True)
            text = json.dumps({"id": 7, "result": {"text": "legal\u2028and\u2029separators"}}, ensure_ascii=False)
            response._content = boundary.join([
                ": heartbeat", "data: {\"method\": \"notifications/progress\"}", "",
                "data: " + text[:9], "data: " + text[9:], "", ""]).encode()
            self.assertEqual(search._parallel_rpc(response, 7)["text"], "legal\u2028and\u2029separators")
        with self.assertRaisesRegex(search.SearchError, "no matching"):
            search._parallel_rpc(reply({}, request_id=9), 1)
        response = reply()
        response._content = b'{"id":1,"error":{"code":-1,"message":"denied"}}'
        with self.assertRaisesRegex(search.SearchError, "denied"):
            search._parallel_rpc(response, 1)

    def test_redirects_and_network_failure_are_not_retried(self):
        response = reply(status=307)
        response.headers["Location"] = "https://example.com/elsewhere"
        with patch("requests.Session.send", return_value=response) as send:
            with self.assertRaisesRegex(search.SearchError, "redirect"):
                search.search_sync("q", provider="parallel")
            send.assert_called_once()
        with patch("requests.Session.send", side_effect=requests.Timeout("synthetic timeout")) as send:
            with self.assertRaisesRegex(search.SearchError, "timeout"):
                search.search_sync("q", provider="parallel")
            send.assert_called_once()

    def test_discovery_protocol_and_payload_failures(self):
        for phase in ("version", "discovery", "payload"):
            responses = exchange()
            if phase == "version":
                responses[0]._content = json.dumps({"id": 1, "result": {
                    "protocolVersion": "unsupported"}}).encode()
                responses[0].headers["Content-Type"] = "application/json"
                responses = [responses[0], responses[-1]]
            elif phase == "discovery":
                responses = responses[:2] + [reply({"tools": []}, 2), responses[-1]]
            else:
                responses[3]._content = b"data: invalid JSON\r\n\r\n"
            with self.subTest(phase=phase), patch("requests.Session.send", side_effect=responses), \
                    patch.object(search, "_search_duckduckgo") as ddg:
                with self.assertRaises(search.SearchError):
                    search.search_sync("q", provider="parallel")
                ddg.assert_not_called()

    def test_upper_limit_and_stateless_server(self):
        responses = exchange()
        responses[0].headers.pop("Mcp-Session-Id")
        with patch("requests.Session.send", side_effect=responses[:4]) as send:
            result = search.search_sync("q", 25, "parallel")
            self.assertEqual(len(result.results), 3)
            self.assertEqual(send.call_count, 4)  # no session to delete

    def test_service_does_not_import_server_frameworks(self):
        subprocess.run([sys.executable, "-c", "from services import search; import sys; "
                        "assert not {'fastmcp', 'fastapi', 'streamlit'} & sys.modules.keys()"], check=True)


if __name__ == "__main__":
    unittest.main()
