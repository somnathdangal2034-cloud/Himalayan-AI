
"""Himalayan AI - Python server using the Google Gemini API."""

import json
import os
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib import request, error

ROOT = Path(__file__).parent.resolve()
PUBLIC = ROOT / "public"

SYSTEM_PROMPT = (
    "You are Himalayan AI, a friendly and knowledgeable assistant. "
    "Give clear, accurate, concise answers. Use plain text; put code inside "
    "triple-backtick fences with the language name."
)


def load_env():
    """Load .env locally. Cloud Run uses environment variables directly."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()

API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODELS = [
    m for m in [
        os.environ.get("GEMINI_MODEL", ""),
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-3.7-flash",
    ] if m
]

# Cloud Run supplies PORT automatically.
PORT = int(os.environ.get("PORT", "8080"))

TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def call_gemini(contents):
    """Returns (status, payload_dict)."""

    last = "The AI service returned an error."

    for model in MODELS:
        body = json.dumps({
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": contents,
        }).encode("utf-8")

        req = request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": API_KEY,
            },
        )

        try:
            with request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())

            candidates = data.get("candidates") or []
            parts = (
                candidates[0].get("content", {}).get("parts", [])
                if candidates else []
            )

            reply = "".join(
                part.get("text", "") for part in parts
            ).strip()

            if reply:
                return 200, {"reply": reply}

            if data.get("promptFeedback", {}).get("blockReason"):
                return 200, {
                    "reply": "I can't answer that one. Try rewording your question."
                }

            return 200, {
                "reply": "I couldn't generate a reply. Please try again."
            }

        except error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get(
                    "error", {}
                ).get("message", "")
            except Exception:
                detail = ""

            print("Gemini API error:", model, e.code, detail)

            if e.code == 404:
                last = "Model not found."
                continue

            if e.code in (400, 403) and "api key" in detail.lower():
                return 502, {
                    "error": "The API key is invalid. Check GEMINI_API_KEY."
                }

            if e.code == 429:
                return 502, {
                    "error": "Free limit reached for now. Wait a minute and try again."
                }

            return 502, {"error": detail or last}

        except Exception as e:
            print("Request failed:", e)
            return 500, {
                "error": "Could not reach the AI service. Check the server connection."
            }

    return 502, {
        "error": "No working Gemini model found. Check GEMINI_MODEL."
    }


class Handler(BaseHTTPRequestHandler):

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type", "application/json; charset=utf-8"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            path = "/index.html"

        target = (PUBLIC / path.lstrip("/")).resolve()

        try:
            target.relative_to(PUBLIC)
        except ValueError:
            return self.send_json(403, {"error": "Forbidden"})

        if not target.is_file():
            return self.send_json(404, {"error": "Not found"})

        data = target.read_bytes()

        self.send_response(200)
        self.send_header(
            "Content-Type",
            TYPES.get(
                target.suffix.lower(),
                "application/octet-stream"
            )
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_json(404, {"error": "Not found"})

        if not API_KEY:
            return self.send_json(500, {
                "error": "Server has no GEMINI_API_KEY environment variable."
            })

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.send_json(400, {"error": "Invalid request size."})

        if length <= 0 or length > 1_000_000:
            return self.send_json(400, {
                "error": "Invalid request size."
            })

        try:
            payload = json.loads(self.rfile.read(length))
            messages = payload.get("messages", [])

            if not isinstance(messages, list):
                return self.send_json(400, {
                    "error": "Invalid messages."
                })

        except (ValueError, AttributeError):
            return self.send_json(400, {
                "error": "Invalid JSON."
            })

        clean = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant"):
                continue

            if not isinstance(content, str) or not content.strip():
                continue

            clean.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content[:40000]}],
            })

        clean = clean[-30:]

        if not clean or clean[-1]["role"] != "user":
            return self.send_json(400, {
                "error": "The last message must come from the user."
            })

        status, response_data = call_gemini(clean)
        self.send_json(status, response_data)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":

    if not API_KEY:
        print("Warning: GEMINI_API_KEY is not set.")

    print(f"Himalayan AI running on port {PORT}")

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        Handler
    )

    server.serve_forever()
