## Train the completion network with graph_outnet
python completion.py  \
       --config configs/dfaust.yaml  \
       SOLVER.alias d7

python completion.py  \
       --config configs/dfaust_d8.yaml  \
       SOLVER.alias d8


## Generate shapes with trained weights
name=0115_ep600_basic_b8_d8_reg_v1
python dualocnn.py  \
       --config configs/dfaust_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 0, \
       SOLVER.resolution 256 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.alias ${name}_time  \
       SOLVER.save_sdf False \
       SOLVER.eval_step  -1

name=0111_ep600_basic_b8_d8
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       DATA.test.filelist data/dfaust/cmp.txt \
       SOLVER.gpu 0, \
       SOLVER.resolution 420 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.alias ${name}_cmp  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  2


## Rename meshes
python tools/rename_meshes.py  \
       --filelist data/dfaust/test_all.txt  \
       --folder_in logs/dfaust_eval/dfaust_${name}  \
       --folder_out logs/dfaust_eval/dfaust_${name}_mesh


## Rescale meshes
python tools/process_dfaust.py  \
       --mesh_folder   logs/dfaust_eval/dfaust_${name}_mesh/meshes \
       --output_folder logs/dfaust_eval/dfaust_${name}_mesh/meshes_rescale \
       --points_folder data/dfaust/dfaust/scans_npy_200k  \
       --filelist data/dfaust/dfaust/test_all.txt  \
       --scale 0.8  \
       --run rescale_mesh


## Flatten folders
python tools/flatten_folders.py  \
       --filelist data/dfaust/dfaust/test_all.txt  \
       --folder_in  logs/dfaust_eval/dfaust_${name}_mesh/meshes_rescale  \
       --folder_out logs/dfaust_eval/dfaust_${name}_mesh/meshes_flatten  


## Compute metrics
python tools/compute_metrics.py  \
       --filelist data/dfaust/dfaust/test_all.txt  \
       --ref_folder data/dfaust/dfaust/mesh_gt  \
       --mesh_folder logs/dfaust_eval/dfaust_${name}_mesh/meshes_rescale  \
       --filename_out logs/dfaust_eval/dfaust_${name}_mesh/metrics.csv


## Visualize SDFs
python tools/visualize_sdf.py  \
       --filename logs/dfaust_eval/dfaust_${name}/0000.sdf.npy
