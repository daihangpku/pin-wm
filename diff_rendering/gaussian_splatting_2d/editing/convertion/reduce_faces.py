
import argparse

def reduce_mesh(input_path: str, output_path: str, target_num_faces: int, backend="pymeshlab"):
    if backend == "pymeshlab":
        import pymeshlab
        """Get a Mesh from a filename."""
        ms = pymeshlab.MeshSet()
        print(f"Loading {input_path}...")
        ms.load_new_mesh(input_path)
        print("Reducing...")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces)
        print("Done.")
        ms.save_current_mesh(output_path)
    elif backend == "trimesh":
        import trimesh
        print(f"Loading {input_path}...")
        mesh = trimesh.load(input_path)
        print("Reducing...")
        simplified = mesh.simplify_quadric_decimation(face_count=target_num_faces)
        print("Done.")
        simplified.export(output_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--target_num_faces', type=int, default=50000)
    parser.add_argument('--backend', type=str, default="pymeshlab")
    args = parser.parse_args()
    reduce_mesh(args.input_path, args.output_path, args.target_num_faces, args.backend)