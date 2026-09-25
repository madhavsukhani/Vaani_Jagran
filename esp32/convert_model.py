#!/usr/bin/env python3
"""Convert a TensorFlow Lite model into the ESP32 model_data.h header."""

from __future__ import annotations

import os
import sys


DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),
    "VAANI_JAGRAN_ASR",
    "model.tflite",
)


def convert_tflite_to_header(tflite_path: str, header_path: str | None = None) -> None:
    """Convert a .tflite file into the symbols expected by the firmware."""
    if not os.path.exists(tflite_path):
        print(f"Error: model file not found: {tflite_path}")
        raise SystemExit(1)

    if header_path is None:
        header_path = os.path.join(os.path.dirname(tflite_path), "model_data.h")

    with open(tflite_path, "rb") as model_file:
        data = model_file.read()

    file_size = len(data)
    print(f"Reading '{tflite_path}' ({file_size} bytes)...")

    with open(header_path, "w", encoding="ascii") as header_file:
        header_file.write(
            "// Auto-generated from "
            f"{os.path.basename(tflite_path)} ({file_size:,} bytes)\n"
        )
        header_file.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        header_file.write("#include <stddef.h>\n#include <stdint.h>\n\n")
        header_file.write("alignas(16) const unsigned char g_model[] = {\n")

        for offset in range(0, file_size, 12):
            chunk = data[offset:offset + 12]
            values = ", ".join(f"0x{byte:02x}" for byte in chunk)
            suffix = "," if offset + 12 < file_size else ""
            header_file.write(f"  {values}{suffix}\n")

        header_file.write("};\n\n")
        header_file.write(f"const int g_model_len = {file_size};\n\n")
        header_file.write("#endif\n")

    print(f"Generated header: {header_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        if os.path.exists(DEFAULT_MODEL):
            convert_tflite_to_header(DEFAULT_MODEL)
        else:
            print("Usage: python3 esp32/convert_model.py path/to/model.tflite")
            raise SystemExit(1)
    else:
        output = sys.argv[2] if len(sys.argv) > 2 else None
        convert_tflite_to_header(sys.argv[1], output)
