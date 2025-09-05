#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3dgs_to_thin_slabs.py
Convert 3D Gaussian Splatting to thin 3DGS slabs by adding thickness dimension.

Input PLY: Standard 3DGS format with fields:
  x,y,z: 3D position
  scale_0,scale_1: existing log-scales  
  rot_0,rot_1,rot_2,rot_3: rotation quaternion
  opacity: opacity
  f_dc_0,f_dc_1,f_dc_2: spherical harmonics coefficients

Output: Same format with added scale_2 for thickness

Usage example:
  python 3dgs_to_thin_slabs.py --input input.ply --output output.ply --sigma-z 0.001

Author: ChatGPT
"""
import argparse
import math
from typing import List
import numpy as np
import struct
import sys

# ---------------------- PLY reader ----------------------

def read_ply(path: str) -> List[dict]:
    """Read PLY file and return list of dictionaries with vertex properties."""
    records = []
    
    with open(path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Parse header to get property names and count
        vertex_count = 0
        property_names = []
        property_types = []
        
        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                property_names.append(prop_name)
                property_types.append(prop_type)
        
        # Read binary data
        if vertex_count > 0:
            # Construct format string for struct.unpack
            fmt = '<'  # little endian
            for prop_type in property_types:
                if prop_type == 'float':
                    fmt += 'f'
                elif prop_type == 'double':
                    fmt += 'd'
                elif prop_type in ['int', 'int32']:
                    fmt += 'i'
                elif prop_type in ['uint', 'uint32']:
                    fmt += 'I'
                elif prop_type in ['short', 'int16']:
                    fmt += 'h'
                elif prop_type in ['ushort', 'uint16']:
                    fmt += 'H'
                elif prop_type in ['char', 'int8']:
                    fmt += 'b'
                elif prop_type in ['uchar', 'uint8']:
                    fmt += 'B'
                else:
                    fmt += 'f'  # default to float
            
            record_size = struct.calcsize(fmt)
            
            for i in range(vertex_count):
                data = f.read(record_size)
                if len(data) != record_size:
                    break
                
                values = struct.unpack(fmt, data)
                record = dict(zip(property_names, values))
                records.append(record)
    
    return records

# ---------------------- PLY writer ----------------------

def write_ply(path: str, records: List[dict]):
    if not records:
        return
    
    # Get all field names from the first record, preserve order
    sample_record = records[0]
    
    # Standard fields in specific order
    standard_fields = ["x","y","z","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3","opacity"]
    
    # Color/SH fields - get all f_dc and f_rest fields in order
    color_fields = []
    for key in sorted(sample_record.keys()):
        if key.startswith("f_dc_") or key.startswith("f_rest_"):
            color_fields.append(key)
    
    # Other fields
    other_fields = []
    for key in sample_record.keys():
        if key not in standard_fields and not key.startswith("f_dc_") and not key.startswith("f_rest_"):
            other_fields.append(key)
    
    # Combine all fields
    names = standard_fields + color_fields + other_fields
    
    # Filter out fields that don't exist in the record
    names = [n for n in names if n in sample_record]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(records)}",
    ]
    # property types
    for n in names:
        header.append(f"property float {n}")
    header.append("end_header\n")
    header_bin = ("\n".join(header)).encode("ascii")

    fmt = "<" + "f"*len(names)
    with open(path, "wb") as f:
        f.write(header_bin)
        for rec in records:
            row = [float(rec[n]) for n in names]
            f.write(struct.pack(fmt, *row))

# ---------------------- Main conversion ----------------------

def convert_3dgs_to_thin_slabs(row, sigma_z=None, sigma_z_rel=None):
    """Convert 3DGS to thin slabs by adding thickness dimension."""
    # Parse 3D position
    x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
    
    # Check if it has the standard 3DGS format
    required_fields = ["scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    if all(field in row for field in required_fields):
        # Get existing scales and add thickness as scale_2
        scale_0 = float(row["scale_0"])
        scale_1 = float(row["scale_1"])
        
        # Determine thickness
        if sigma_z is not None:
            sz = sigma_z
        elif sigma_z_rel is not None:
            sz = max(abs(z) * sigma_z_rel, 1e-6)
        else:
            sz = 0.002 * max(1.0, abs(z))  # sensible default
        
        # Convert to log scale (3DGS expects log-scales)
        scale_2 = math.log(max(sz, 1e-9))
        
        # Start with required fields
        result = dict(
            x=x, y=y, z=z,
            scale_0=scale_0, scale_1=scale_1, scale_2=scale_2,
            rot_0=float(row["rot_0"]), rot_1=float(row["rot_1"]), 
            rot_2=float(row["rot_2"]), rot_3=float(row["rot_3"]),
            opacity=float(row["opacity"]),
        )
        
        # Copy all color/SH coefficients (f_dc_* and f_rest_*)
        for key, value in row.items():
            if key.startswith("f_dc_") or key.startswith("f_rest_"):
                result[key] = float(value)
        
        # Copy any other fields (like normals nx, ny, nz)
        for key, value in row.items():
            if key not in result and key not in ["x", "y", "z", "scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3", "opacity"]:
                result[key] = float(value)
        
        return result
    else:
        print(f"Error: Input file is not in standard 3DGS format. Missing fields: {[f for f in required_fields if f not in row]}")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Convert 3DGS to thin slabs (.ply).")
    ap.add_argument("--input", type=str, required=True, help="Input PLY file")
    ap.add_argument("--output", type=str, required=True, help="Output PLY path")
    ap.add_argument("--sigma-z", dest="sigma_z", type=float, default=None, help="Absolute slab thickness std (meters)")
    ap.add_argument("--sigma-z-rel", dest="sigma_z_rel", type=float, default=None, help="Relative thickness (std = rel * depth)")
    args = ap.parse_args()

    recs = []
    records = read_ply(args.input)
    
    # Debug: print available fields in PLY file
    if records:
        print(f"PLY file contains {len(records)} records")
        print(f"Available fields: {list(records[0].keys())}")
        print("Converting 3DGS to thin slabs...")
    
    for row in records:
        rec = convert_3dgs_to_thin_slabs(row, args.sigma_z, args.sigma_z_rel)
        recs.append(rec)

    if not recs:
        print("No rows parsed. Check your PLY file.", file=sys.stderr)
        sys.exit(1)

    write_ply(args.output, recs)
    print(f"Wrote {len(recs)} thin slab Gaussians to {args.output}")

if __name__ == "__main__":
    main()
