"""Small HTTP fixture for verifying the public benchmark entry and its streaming client."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sys


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"fixture"}]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert body["max_tokens"] == 256 and body["stream"]
        assert self.headers["x-session-id"] == self.headers["x-delivery-request-id"] == body["cache_salt"]
        event = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                 "choices": [{"delta": {"content": "Answer: A"}, "finish_reason": "stop"}]}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
