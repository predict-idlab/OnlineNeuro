# /networking/data_transfer.py
import json
import select
import socket
import struct
from typing import Any, Dict, Optional


def _read_n_bytes(sock: socket.socket, n: int) -> Optional[bytes]:
    """Helper function to reliably read exactly n bytes from a socket."""
    buffer = bytearray()
    while len(buffer) < n:
        try:
            # The amount to read is the remaining number of bytes
            part = sock.recv(n - len(buffer))
            if not part:
                # Socket was closed by the peer
                return None
            buffer.extend(part)
        except BlockingIOError:
            # This can happen with non-blocking sockets, but we'll use select
            # to wait, so it's less of an issue. Still, good practice to handle.
            continue
        except ConnectionError:
            return None
    return bytes(buffer)


def send_message(sock: socket.socket, data: Any) -> Dict[str, Any]:
    """
    Serializes and sends a Python object over a socket using length-prefix framing.

    Args:
        sock: The active socket to send data to.
        data: Any JSON-serializable Python object (dict, list, etc.).

    Returns:
        A dictionary indicating the status of the operation.
    """
    try:
        # 1. Serialize the data to a JSON string, then encode to UTF-8 bytes.
        message_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")

        # 2. Pack the length of the message into a 4-byte, big-endian, unsigned integer header.
        header = struct.pack("!I", len(message_bytes))

        # 3. Send the header followed by the message bytes.
        sock.sendall(header + message_bytes)

        return {"status": "success", "bytes_sent": len(header) + len(message_bytes)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def receive_message(sock: socket.socket, timeout: float = 300.0) -> Optional[Any]:
    """
    Receives and deserializes a Python object from a socket using length-prefix framing.

    Args:
        sock: The active socket to receive data from.
        timeout: The total time in seconds to wait for a complete message.

    Returns:
        The deserialized Python object, or None if the connection is closed,
        an error occurs, or the timeout is reached.

    TODO
    ----
        This should always return a dictionary, if required, add an error message
        for better handling
    """
    try:
        # Use select to wait for data to become available, respecting the timeout
        ready_to_read, _, _ = select.select([sock], [], [], timeout)

        if not ready_to_read:
            print("Timeout: No data received from socket within the timeout period.")
            return None

        # 1. Read the 4-byte header to get the message length.
        header_bytes = _read_n_bytes(sock, 4)
        if header_bytes is None:
            print("Connection closed while reading message header.")
            return None

        message_length = struct.unpack("!I", header_bytes)[0]

        # Sanity check for extremely large messages to prevent memory issues
        # (e.g., 1GB limit). Adjust as needed.
        if message_length > 1_000_000_000:
            raise ValueError(f"Message size {message_length} is too large.")

        # 2. Read the full message body based on the determined length.
        message_bytes = _read_n_bytes(sock, message_length)
        if message_bytes is None:
            print("Connection closed while reading message body.")
            return None

        # 3. Decode from UTF-8 bytes to string, then deserialize from JSON.
        message = json.loads(message_bytes.decode("utf-8"))

        return message

    except (ValueError, struct.error, json.JSONDecodeError, ConnectionError) as e:
        print(f"Error receiving message: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred during receive: {e}")
        return None
