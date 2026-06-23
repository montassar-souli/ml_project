"""Minimal HTTP server for the frontend UI on port 8000."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os

from app import build_ui_html


class FrontendHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		if self.path == "/health":
			payload = b"ok"
			self.send_response(200)
			self.send_header("Content-Type", "text/plain; charset=utf-8")
			self.send_header("Content-Length", str(len(payload)))
			self.end_headers()
			self.wfile.write(payload)
			return

		html = build_ui_html(os.getenv("API_BASE_URL", "http://localhost:5000"))
		payload = html.encode("utf-8")
		self.send_response(200)
		self.send_header("Content-Type", "text/html; charset=utf-8")
		self.send_header("Content-Length", str(len(payload)))
		self.end_headers()
		self.wfile.write(payload)

	def log_message(self, format, *args):
		return


def main() -> None:
	port = int(os.getenv("FRONTEND_PORT", "8000"))
	server = ThreadingHTTPServer(("0.0.0.0", port), FrontendHandler)
	print(f"[FRONT] serving on 0.0.0.0:{port}")
	server.serve_forever()


if __name__ == "__main__":
	main()