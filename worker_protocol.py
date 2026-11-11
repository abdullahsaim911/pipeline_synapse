"""Small framed stdio protocol shared by owned local model workers."""

from __future__ import annotations

import pickle
import struct
from typing import Any


def read_frame(stream) -> Any:
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError
    size = struct.unpack("!I", header)[0]
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError
    return pickle.loads(payload)


def write_frame(stream, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    stream.flush()
