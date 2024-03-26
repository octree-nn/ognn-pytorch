## Train the completion network with graph_outnet
python completion.py  \
       --config configs/synthetic_room.yaml  \
       SOLVER.alias reg


## Test the completion network with graph_outnet
name=gpu4_ep600_poly_d7_basic_reweight_1
python completion.py  \
       --config configs/synthetic_room_eval.yaml  \
       MODEL.name graph_ounet  \
       DATA.test.name synthetic_room  \
       DATA.test.location data/synthetic_room_dataset  \
       DATA.test.filelist data/synthetic_room_dataset/filelist/all_test.txt  \
       SOLVER.gpu 1,  \
       SOLVER.resolution 256 \
       SOLVER.save_sdf True \
       SOLVER.ckpt logs/room/${name}/checkpoints/00600.model.pth  \
       SOLVER.alias ${name} \
       SOLVER.eval_step 5


## Generate shapes with trained weights
name=1218_ep900_poly_d7_basic_cls
python completion.py  \
      --config configs/synthetic_room_eval.yaml  \
      MODEL.name graph_ounet  \
      MODEL.resblk_type basic \
      SOLVER.gpu 0, \
      SOLVER.resolution 200 \
      SOLVER.sdf_scale 0.9  \
      SOLVER.ckpt logs/room/${name}/checkpoints/00900.model.pth   \
      SOLVER.alias ${name}  \
      SOLVER.save_sdf True \
      SOLVER.eval_step 5


## Rename meshes
python tools/rename_meshes.py   \
       --filelist data/synthetic_room_dataset/filelist/all_test.txt   \
       --folder_in logs/room_eval/all_${name}  \
       --folder_out logs/room_eval/all_${name}_mesh


## Visualize SDFs
python tools/visualize_sdf.py  \
       --filename logs/room_eval/all_${name}/0000.sdf.npy \
       --sdf_scale 0.9 \
       --mesh_scale 0.6


## Evaluate meshes
python eval_meshes.py  \
       configs/pointcloud/pretrained/room_grid64.yaml  \
       --suffix obj   \
       --generation_dir /mnt/logs/docnn/room_eval/all_${name}_mesh
