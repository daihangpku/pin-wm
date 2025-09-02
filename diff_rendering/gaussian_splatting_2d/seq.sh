object_name=banana

python undistort.py --dataset_dir ./datasets/$object_name --test_ratio 0.1
python train.py -s ./datasets/$object_name --model_path ./output/$object_name
python render.py --model_path ./output/$object_name