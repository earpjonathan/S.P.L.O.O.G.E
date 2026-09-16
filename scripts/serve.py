#!/usr/bin/env python
"""Static server for viewer/ that supports HTTP Range requests.

python -m http.server does NOT implement Range: it answers every request with
200 and the whole file. Splats do not care (they are streamed with fetch and
read start-to-end), but <video> does: without 206 support the browser can only
seek inside whatever it has already buffered, and a seek past that silently
snaps currentTime back to 0. That looked exactly like a broken sync bug in the
footage overlay.
"""
import http.server, os, re, socketserver, sys

ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/Desktop/fpv-splat/viewer')
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8777


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def send_head(self):
        rng = self.headers.get('Range')
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404)
            return None
        size = os.fstat(f.fileno()).st_size
        m = re.match(r'bytes=(\d*)-(\d*)$', rng.strip())
        if not m:
            f.close()
            self.send_error(400, 'Bad Range')
            return None
        a, b = m.group(1), m.group(2)
        if a:
            start = int(a)
            end = int(b) if b else size - 1
        else:                        # suffix form: bytes=-N  (last N bytes)
            if not b:
                f.close()
                self.send_error(400, 'Bad Range')
                return None
            start = max(0, size - int(b))
            end = size - 1
        if start >= size:
            f.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()
            return None
        end = min(end, size - 1)
        f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        self._range = (start, end)
        return f

    def copyfile(self, src, dst):
        if not hasattr(self, '_range'):
            return super().copyfile(src, dst)
        start, end = self._range
        del self._range
        remaining = end - start + 1
        while remaining > 0:
            chunk = src.read(min(64 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        # so a plain 200 also advertises that seeking is possible
        if 'Accept-Ranges' not in self._headers_buffer_str():
            self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def _headers_buffer_str(self):
        return b''.join(getattr(self, '_headers_buffer', []) or []).decode('latin-1')

    def log_message(self, *a):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    with Server(('127.0.0.1', PORT), RangeHandler) as httpd:
        print(f'serving {ROOT} on http://127.0.0.1:{PORT} (Range supported)')
        httpd.serve_forever()
