import argparse
import json
import socket
import socketserver


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        chunks = []
        while True:
            data = self.request.recv(4096)
            if not data:
                break
            chunks.append(data)
            if len(data) < 4096:
                break

        payload = b"".join(chunks)
        response = json.dumps(
            {
                "status": "ok",
                "received_size": len(payload),
            }
        ).encode("utf-8")
        self.request.sendall(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    class ThreadedTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with ThreadedTCPServer((args.bind, args.port), EchoHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
