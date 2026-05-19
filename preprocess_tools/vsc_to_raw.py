"""
VSC → RAW converter for VisualScan 2011 ultrasound data.

The .vsc format is GZIP-compressed .NET BinaryFormatter data produced by
the VisualScan 2011 machine software.  Internally it stores:
  - Version / path / date strings
  - Scan dimensions X, Y, Z as System.Int32 values
  - A 2D C-scan array (X*Y bytes, amplitude map)
  - Y scan-line arrays, each Z*X bytes (one per Y position)

On disk, the data is arranged as (Y, Z, X).  The standard .raw format used
by the rest of this pipeline is (Z, Y, X) in numpy / C order, matching the
widthxheightxdepth convention in filenames (i.e. _ascan{X}x{Y}x{Z}.raw).
"""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np

_VSC_MARKER = b"Fichero VisualSCan2011"
_DIM_SIGNATURE = b"m_value\x00\x08"   # appears before each System.Int32 value
_ARRAY_PRIMITIVE_BYTE = 0x0F          # ArraySinglePrimitive record type
_PRIM_TYPE_BYTE = 0x02                # PrimitiveTypeEnumeration: Byte


def _parse_header(data: bytes) -> tuple[int, int, int]:
    """Extract scan dimensions (X, Y, Z) from decompressed VSC bytes."""
    if _VSC_MARKER not in data[:300]:
        raise ValueError(
            "Not a valid VisualSCan2011 file — version marker not found."
        )
    dims: list[int] = []
    pos = 0
    while len(dims) < 3:
        p = data.find(_DIM_SIGNATURE, pos, 2000)
        if p < 0:
            break
        dims.append(struct.unpack_from("<i", data, p + len(_DIM_SIGNATURE))[0])
        pos = p + 1
    if len(dims) != 3:
        raise ValueError(
            f"Expected 3 dimension values in header, found {len(dims)}."
        )
    return dims[0], dims[1], dims[2]


def _extract_scan_data(data: bytes, X: int, Y: int, Z: int) -> np.ndarray:
    """
    Locate the Y scan-line arrays and assemble them into a (Z, Y, X) volume.

    Each scan line is stored as an ArraySinglePrimitive record of Z*X bytes
    (one per Y position).  Concatenating all Y lines gives a (Y, Z, X) block;
    transposing axes (1, 0, 2) yields the target (Z, Y, X) layout.
    """
    slice_size = X * Z
    slices: list[bytes] = []
    pos = 0
    while pos < len(data) - 10:
        if data[pos] == _ARRAY_PRIMITIVE_BYTE:
            length = struct.unpack_from("<i", data, pos + 5)[0]
            if data[pos + 9] == _PRIM_TYPE_BYTE and length == slice_size:
                start = pos + 10
                slices.append(data[start: start + length])
        pos += 1

    if len(slices) != Y:
        raise ValueError(
            f"Expected {Y} scan-line arrays of size {slice_size}, "
            f"found {len(slices)}."
        )

    raw = np.frombuffer(b"".join(slices), dtype=np.uint8)
    return raw.reshape(Y, Z, X).transpose(1, 0, 2)   # → (Z, Y, X)


def vsc_to_raw(
    vsc_path: str | Path,
    output_dir: str | Path | None = None,
    spacing_xy: float = 1.0,
    spacing_z: float = 0.02232141,
    velocity: int = 2970,
) -> dict:
    """
    Convert a VisualScan .vsc file to a flat .raw volume and companion info file.

    Parameters
    ----------
    vsc_path : path-like
        Path to the input .vsc file.
    output_dir : path-like, optional
        Directory where output files are written.  Defaults to the same
        directory as the input file.
    spacing_xy : float
        Scan step in mm for X and Y axes.  Default: 1.0 mm.
    spacing_z : float
        Sample spacing in mm along the Z (depth) axis.  Default: 0.02232141 mm.
    velocity : int
        Ultrasonic propagation velocity in m/s.  Default: 2970 m/s.

    Returns
    -------
    dict with keys:
        'raw'   : Path to the written .raw file.
        'info'  : Path to the written _ascan_info.txt file.
        'shape' : (X, Y, Z) tuple of scan dimensions.
    """
    vsc_path = Path(vsc_path)
    if output_dir is None:
        out_dir = vsc_path.parent
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(vsc_path, "rb") as fh:
        data = fh.read()

    X, Y, Z = _parse_header(data)
    volume = _extract_scan_data(data, X, Y, Z)   # shape (Z, Y, X), dtype uint8

    stem = vsc_path.stem
    raw_name = f"{stem}_ascan{X}x{Y}x{Z}.raw"
    info_name = f"{stem}_ascan_info.txt"

    raw_path = out_dir / raw_name
    info_path = out_dir / info_name

    volume.tofile(raw_path)

    info_path.write_text(
        f"Dimension X: {X} mm\n"
        f"Dimension Y: {Y} mm\n"
        f"Dimension Z: {Z} mm\n"
        f"Distancia entre dos puntos X: {spacing_xy} mm\n"
        f"Distancia entre dos puntos Y: {spacing_xy} mm\n"
        f"Distancia entre dos puntos Z: {spacing_z} mm\n"
        f"Velocidad de propagacion del material:{velocity}m/s\n",
        encoding="utf-8",
    )

    return {"raw": raw_path, "info": info_path, "shape": (X, Y, Z)}


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a VisualScan .vsc file to .raw + _ascan_info.txt."
    )
    parser.add_argument("vsc_file", type=Path, help="Input .vsc file")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--spacing-xy", type=float, default=1.0,
        help="Scan step in mm for X and Y (default: 1.0)",
    )
    parser.add_argument(
        "--spacing-z", type=float, default=0.02232141,
        help="Sample spacing in mm for Z (default: 0.02232141)",
    )
    parser.add_argument(
        "--velocity", type=int, default=2970,
        help="Propagation velocity in m/s (default: 2970)",
    )
    args = parser.parse_args()

    result = vsc_to_raw(
        args.vsc_file,
        output_dir=args.output_dir,
        spacing_xy=args.spacing_xy,
        spacing_z=args.spacing_z,
        velocity=args.velocity,
    )
    X, Y, Z = result["shape"]
    print(f"Shape : {X}x{Y}x{Z}")
    print(f"RAW   : {result['raw']}")
    print(f"Info  : {result['info']}")


if __name__ == "__main__":
    _main()
