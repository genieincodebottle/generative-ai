"""Synthetic local HTTP fixtures, not recorded Parallel responses.

Flush a complete reply and hold HTTP open until the client returns. A short
event deadline detects buffering without waiting for a network timeout.
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import requests
from services import search


class OpenStreamTests(unittest.TestCase):
    def check_open_stream(self, content_type, boundary, rpc_error=False):
        flushed, returned, release = (threading.Event() for _ in range(3))
        outcome, closed = [], []
        text = "Evidence with café\u2028and\u2029Unicode."

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                method = request['method']
                if method == 'notifications/initialized':
                    self.send_response(202)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if method == 'initialize':
                    result = {'protocolVersion': '2025-03-26'}
                elif method == 'tools/list':
                    result = {'tools': [{'name': 'web_search'}]}
                else:
                    result = {'structuredContent': {'results': [{
                        'url': 'https://example.com/source', 'title': 'Synthetic source',
                        'excerpts': [text]}]}}
                message = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
                if method == 'tools/call' and rpc_error:
                    message.pop('result')
                    message['error'] = {'code': -1, 'message': 'synthetic denied'}
                if method != 'tools/call':
                    payload = json.dumps(message).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Connection', 'close')
                self.end_headers()
                lines = [': heartbeat', 'event: ping', 'data: keepalive', '', 'data: {"method":"notifications/progress"}', '',
                         'data: {"id":999,"result":{}}', '', 'event: message',
                         'data: ' + json.dumps(message, ensure_ascii=False), '', '']
                self.wfile.write(boundary.join(lines).encode('utf-8'))
                self.wfile.flush()
                flushed.set()
                release.wait(3)
                self.close_connection = True

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        def call():
            try:
                outcome.append(search.search_sync('synthetic query', 1, 'parallel'))
            except Exception as exc:
                outcome.append(exc)
            finally:
                returned.set()

        original_close = requests.Response.close

        def close(response):
            result = original_close(response)
            if response.headers.get('Content-Type') == content_type:
                closed.append(response.raw.closed)
            return result

        client = threading.Thread(target=call, daemon=True)
        try:
            with patch.object(search, 'PARALLEL_MCP_URL', f'http://127.0.0.1:{server.server_port}/mcp'), \
                    patch.dict('os.environ', {}, clear=True), \
                    patch.object(requests.Response, 'close', close):
                client.start()
                self.assertTrue(flushed.wait(2), 'fixture did not flush a reply')
                self.assertTrue(returned.wait(1), 'client waited for EOF after a complete RPC reply')
                self.assertFalse(release.is_set(), 'fixture must still be held open')
                self.assertTrue(closed and all(closed), 'client must close the matched response')
                if rpc_error:
                    self.assertIsInstance(outcome[0], search.SearchError)
                    self.assertIn('synthetic denied', str(outcome[0]))
                else:
                    self.assertIsInstance(outcome[0], search.SearchResponse)
                    self.assertEqual(outcome[0].results[0].snippet, text)
        finally:
            release.set()
            client.join(2)
            server.shutdown()
            server.server_close()
            server_thread.join(2)

    def test_still_open_sse_result(self):
        self.check_open_stream('text/event-stream', '\r\n')

    def test_still_open_sse_rpc_error(self):
        self.check_open_stream('text/event-stream', '\n', rpc_error=True)

    def test_still_open_mixed_case_sse(self):
        for rpc_error in (False, True):
            with self.subTest(rpc_error=rpc_error):
                self.check_open_stream('Text/Event-Stream ; Charset=UTF-8', '\r', rpc_error)
