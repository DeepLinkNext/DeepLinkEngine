"""Check a local HTTP endpoint using the same reuse policy as asyncio."""
import socket


def available(host='127.0.0.1', port=18390):
    with socket.socket() as probe:
        # A closed previous phase may leave accepted connections in TIME_WAIT.
        # This permits their reuse, but still rejects an existing listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
