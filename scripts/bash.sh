
# Single shape
name=0115_ep600_basic_b8_d8_reg_v1
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 0, \
       SOLVER.resolution 320 \
       DATA.test.filelist data/Shapes/filelist_thai_statue.txt \
       DATA.test.location  data/Shapes \
       DATA.test.point_scale 0.004 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.logdir logs/shapes/shapes  \
       SOLVER.alias ${name}_statue_320  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1


python completion.py  \
       --config configs/dfaust_d8.yaml  \
       SOLVER.alias debug_2obj  \
       SOLVER.max_epoch 10000 \
       SOLVER.test_every_epoch 500 \
       DATA.train.batch_size 2 \
       DATA.train.batch_size 2 \
       DATA.train.filelist data/dfaust/train_2.txt \
       DATA.test.filelist data/dfaust/train_2.txt 

name=dfaust_debug_2obj
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 1, \
       SOLVER.resolution 256 \
       DATA.test.filelist data/dfaust/train_2.txt \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/10000.model.pth \
       SOLVER.alias ${name}_2obj_1  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1


name=0115_ep600_basic_b8_d8_reg_v1
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 1, \
       SOLVER.resolution 256 \
       DATA.test.filelist data/SplinePE/bunny.txt \
       DATA.test.location  data/SplinePE/shapes \
       DATA.test.point_scale 0.6 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.alias ${name}_bunny  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1





# ===========================



## Train the completion network with graph_outnet
python completion.py  \
       --config configs/shapenet.yaml  \
       MODEL.name graph_ounet  \
       SOLVER.alias graph_ounet

python completion.py  \
       --config configs/synthetic_room.yaml  \
       SOLVER.alias time

python completion.py  \
       --config configs/dfaust.yaml  \
       SOLVER.alias v0

## Test the completion network with graph_outnet
python completion.py  \
       --config configs/shapenet_eval.yaml  \
       MODEL.name graph_ounet  \
       DATA.test.name shapenet  \
       DATA.test.location data/ShapeNet/dataset  \
       SOLVER.ckpt logs/shapenet/all_graphall_1124/checkpoints/00320.model.pth  \
       SOLVER.alias graph_ounet_1124_test \
       SOLVER.eval_step 5

python completion.py  \
       --config configs/synthetic_room_eval.yaml  \
       MODEL.name graph_ounet  \
       DATA.test.name synthetic_room  \
       DATA.test.location data/synthetic_room_dataset  \
       DATA.test.filelist data/synthetic_room_dataset/filelist/all_train.txt  \
       SOLVER.gpu 1,  \
       SOLVER.ckpt logs/room/all_surf_points/checkpoints/00600.model.pth  \
       SOLVER.alias surf_points_train \
       SOLVER.eval_step 5


python completion.py  \
       --config configs/dfaust_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 0,  \
       SOLVER.ckpt logs/dfaust/dfaust_ref/checkpoints/00040.model.pth \
       SOLVER.alias ref \
       SOLVER.eval_step 5


## Generate shapes with trained weights
name=gpu4_ep300
python completion.py  \
      --config configs/shapenet_eval.yaml  \
      MODEL.name graph_ounet  \
      SOLVER.ckpt logs/shapenet/gpu4_ep300_resblk2/checkpoints/00300.model.pth  \
      SOLVER.alias ${name}  \
      MODEL.resblk_type basic \
      SOLVER.eval_step 5


name=gpu4_ep600
python completion.py  \
      --config configs/synthetic_room_eval.yaml  \
      MODEL.name graph_ounet  \
      MODEL.resblk_type bottleneck \
      SOLVER.gpu 1, \
      SOLVER.ckpt logs/room/gpu4_ep600_poly/checkpoints/00600.model.pth  \
      SOLVER.alias ${name}  \
      SOLVER.eval_step 5


name=gpu4_ep600_d7
python completion.py  \
      --config configs/synthetic_room_eval.yaml  \
      MODEL.name graph_ounet  \
      MODEL.resblk_type bottleneck \
      DATA.test.depth 7 \
      MODEL.depth 7 \
      MODEL.depth_out 7 \
      SOLVER.gpu 1, \
      SOLVER.resolution 256 \
      SOLVER.ckpt logs/room/gpu4_ep600_poly_d7/checkpoints/00600.model.pth  \
      SOLVER.alias ${name}  \
      SOLVER.eval_step 5


python completion.py  \
       --config configs/dfaust_eval.yaml  \
       MODEL.name graph_unet  \
       DATA.test.depth 7  \
       MODEL.depth 7  \
       MODEL.depth_out 7  \
       SOLVER.gpu 0, \
       SOLVER.ckpt logs/dfaust/dfaust_d7/checkpoints/00400.model.pth \
       SOLVER.alias d7_256  \
       SOLVER.resolution 256 \
       SOLVER.eval_step 5 \


## Extract meshes
python tools/marching_cubes.py  \
       --filelist data/ShapeNet/filelist/all_test.txt  \
       --folder_in logs/shapenet_eval/all_${name}  \
       --folder_out logs/shapenet_eval/all_${name}_mesh/meshes

python tools/marching_cubes.py  \
       --filelist data/synthetic_room_dataset/filelist/all_test.txt  \
       --folder_in logs/room_eval/all_${name}  \
       --folder_out logs/room_eval/all_${name}_mesh/meshes  \
       --scale 0.6  \
       --rescale_sdf true


## Visualize SDFs
python tools/visualize_sdf.py  \
       --filename logs/shapenet_eval/all_${name}/0000.sdf.npy

python tools/visualize_sdf.py  \
       --filename logs/room_eval/all_${name}/0000.sdf.npy \
       --rescale_sdf true


## on the dataset of deepmls
python completion.py --config configs/deepmls.yaml


## On the dataset of ConvONet
python regression.py --config configs/noise2clean_occu.yaml
python regression.py --config configs/noise2clean_occu_eval.yaml
