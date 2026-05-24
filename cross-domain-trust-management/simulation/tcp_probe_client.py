import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--payload-size", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--retries", type=int, required=True)
    args = parser.parse_args()

    payload = (b"X" * args.payload_size)
    last_error = None
    timeout_occurred = 0

    for attempt in range(args.retries + 1):
        start = time.perf_counter()
        try:
            with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
                sock.settimeout(args.timeout)
                sock.sendall(payload)
                sock.shutdown(socket.SHUT_WR)

                response = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response += chunk

            elapsed = time.perf_counter() - start
            response_json = json.loads(response.decode("utf-8"))
            payload_ok = response_json.get("received_size") == len(payload)
            throughput_kbps = 0.0
            if elapsed > 0:
                throughput_kbps = round((len(payload) * 8) / elapsed / 1000, 6)

            result = {
                "success": 1 if payload_ok else 0,
                "latency_ms": round(elapsed * 1000, 6),
                "response_size": len(response),
                "connection_error": None if payload_ok else "payload_mismatch",
                "retries": attempt,
                "throughput_kbps": throughput_kbps,
                "payload_ok": payload_ok,
                "timeout_occurred": 0,
            }
            print(json.dumps(result))
            return

        except socket.timeout:
            last_error = "timeout"
            timeout_occurred = 1
        except ConnectionRefusedError:
            last_error = "connection_refused"
        except ConnectionResetError:
            last_error = "connection_reset"
        except Exception as exc:
            last_error = exc.__class__.__name__.lower()

    result = {
        "success": 0,
        "latency_ms": None,
        "response_size": None,
        "connection_error": last_error,
        "retries": args.retries,
        "throughput_kbps": None,
        "payload_ok": False,
        "timeout_occurred": timeout_occurred,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
