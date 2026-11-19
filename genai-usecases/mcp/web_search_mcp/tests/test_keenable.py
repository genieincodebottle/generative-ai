"""Keenable provider tests. All HTTP responses here are synthetic.

Run: python -m unittest discover -s tests -v
"""
import json
import os
import unittest
from unittest.mock import patch

import requests
from services import search


def response(payload=None, status=200, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(payload if payload is not None else {}).encode()
    r.headers["Content-Type"] = "application/json"
    r.headers.update(headers or {})
    return r


RESULTS = {"results": [
    {"title": "One", "url": "https://example.com/1", "snippet": "first"},
    {"title": "", "url": "https://example.com/2", "snippet": None},
    {"title": "No link", "url": "", "snippet": "dropped"},
    {"title": "Bad scheme", "url": "javascript:alert(1)", "snippet": "dropped"},
    "not a dict",
    {"title": "Three", "url": "http://example.com/3", "snippet": "third"},
]}


class KeenableTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_opt_in_only(self):
        self.assertIn("keenable", search.available_providers())
        self.assertEqual(search.default_provider(), "duckduckgo")
        with patch.object(search, "_search_duckduckgo", return_value="ddg"), \
                patch("requests.post") as post:
            self.assertEqual(search.search_sync("question"), "ddg")
            post.assert_not_called()

    def test_keyless_request_and_mapping(self):
        with patch("requests.post", return_value=response(RESULTS)) as post:
            result = search.search_sync("  question  ", 5, "keenable")
        args, kwargs = post.call_args
        self.assertEqual(args[0], search.KEENABLE_PUBLIC_SEARCH_URL)
        self.assertEqual(kwargs["json"], {"query": "question", "max_results": 5})
        self.assertEqual(kwargs["headers"], {"X-Keenable-Title": search.KEENABLE_TITLE})
        self.assertIs(kwargs["allow_redirects"], False)
        self.assertEqual(result.provider, "keenable")
        self.assertEqual([(r.title, r.link, r.snippet) for r in result.results], [
            ("One", "https://example.com/1", "first"),
            ("https://example.com/2", "https://example.com/2", ""),
            ("Three", "http://example.com/3", "third"),
        ])

    def test_key_uses_keyed_endpoint_and_limits(self):
        with patch.dict(os.environ, {"KEENABLE_API_KEY": "secret"}), \
                patch("requests.post", return_value=response(RESULTS)) as post:
            result = search.search_sync("q", 2, "keenable")
        args, kwargs = post.call_args
        self.assertEqual(args[0], search.KEENABLE_SEARCH_URL)
        self.assertEqual(kwargs["headers"]["X-API-Key"], "secret")
        self.assertEqual(len(result.results), 2)
        with patch("requests.post", return_value=response(RESULTS)) as post:
            search.search_sync("q", 25, "keenable")
        self.assertEqual(post.call_args.kwargs["json"]["max_results"], search.KEENABLE_MAX_RESULTS)

    def test_errors_do_not_fall_back(self):
        cases = [
            response(status=403),
            response(status=429),
            response(status=500),
            response(status=302, headers={"Location": "https://elsewhere.example"}),
            requests.ConnectionError("down"),
        ]
        for case in cases:
            kwargs = {"side_effect": case} if isinstance(case, Exception) else {"return_value": case}
            with self.subTest(case=case), patch("requests.post", **kwargs), \
                    patch.object(search, "_search_duckduckgo") as ddg, \
                    self.assertRaises(search.SearchError):
                search.search_sync("q", provider="keenable")
            ddg.assert_not_called()

    def test_empty_and_malformed_payloads(self):
        with patch("requests.post", return_value=response({"results": []})):
            self.assertEqual(search.search_sync("q", provider="keenable").results, [])
        with patch("requests.post", return_value=response({})):
            self.assertEqual(search.search_sync("q", provider="keenable").results, [])
        bad = requests.Response()
        bad.status_code = 200
        bad._content = b"not json"
        with patch("requests.post", return_value=bad), self.assertRaises(search.SearchError):
            search.search_sync("q", provider="keenable")


if __name__ == "__main__":
    unittest.main()
