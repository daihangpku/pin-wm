
output_cache_dir=./output
object_name=small-cup

python editing/annotation/correct_obj_axes.py --input $output_cache_dir/$object_name/train/ours_30000/fuse_post.ply
python editing/convertion/ply2obj.py --input $output_cache_dir/$object_name/train/ours_30000/fuse_post_abs.ply
python editing/convertion/rigid2urdf.py --input $output_cache_dir/$object_name/train/ours_30000/fuse_post_abs.obj
python editing/annotation/correct_3dgs_axes.py \
    --input $output_cache_dir/$object_name/point_cloud/iteration_30000/point_cloud.ply \
    --rel2abs $output_cache_dir/$object_name/train/ours_30000/rel2abs.npz
python editing/convertion/two2three.py \
    --input $output_cache_dir/$object_name/point_cloud/iteration_30000/point_cloud_abs.ply\
    --output $output_cache_dir/$object_name/train/ours_30000/object_3dgs_abs.ply \
    --sigma-z-rel 0.000000001
