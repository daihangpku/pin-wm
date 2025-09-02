
output_cache_dir=./output
object_name=milk-box

python editing/annotation/correct_obj_axes.py --input $output_cache_dir/$object_name/point_cloud/iteration_30000/point_cloud.ply
# python editing/convertion/ply2obj.py --input $output_cache_dir/$object_name/mesh_w_vertex_color_abs.ply
# python editing/convertion/rigid2urdf.py --input $output_cache_dir/$object_name/mesh_w_vertex_color_abs.obj
# python editing/annotation/correct_3dgs_axes.py --input $output_cache_dir/$object_name/object_3dgs.ply