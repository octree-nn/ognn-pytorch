## Train the completion network with graph_outnet
python main.py  \
       --config configs/shapenet.yaml  \
       SOLVER.alias ref

## Test the completion network with graph_outnet
python main.py  \
       --config configs/shapenet_eval.yaml  \
       MODEL.name graph_ounet  \
       DATA.test.name shapenet  \
       DATA.test.location data/ShapeNet/dataset  \
       SOLVER.ckpt logs/shapenet/all_graphall_1124/checkpoints/00320.model.pth  \
       SOLVER.alias graph_ounet_1124_test \
       SOLVER.eval_step 5


## Generate shapes with trained weights
name=1220_ep300_poly_resblk2_weight1
python main.py  \
      --config configs/shapenet_eval.yaml  \
      MODEL.name graph_ounet  \
      MODEL.resblk_type basic \
      SOLVER.ckpt logs/shapenet/${name}/checkpoints/00300.model.pth  \
      SOLVER.alias ${name}  \
      SOLVER.sdf_scale 0.9  \
      SOLVER.save_sdf False \
      SOLVER.eval_step -1

name=1221_ep300_poly_resblk2_octree_ounet
python main.py  \
      --config configs/shapenet_eval.yaml  \
      MODEL.name octree_ounet  \
      SOLVER.ckpt logs/shapenet/${name}/checkpoints/00300.model.pth  \
      SOLVER.alias ${name}  \
      SOLVER.sdf_scale 0.9  \
      SOLVER.save_sdf False \
      SOLVER.eval_step -1


## Evaluate meshes
python eval_meshes.py  \
       configs/pointcloud/shapenet_grid32.yaml  \
       --suffix obj   \
       --generation_dir /mnt/logs/docnn/shapenet_eval/all_${name}_mesh

python eval_meshes.py  \
        configs/pointcloud/shapenet.yaml  \
        --dataset_folder /home/wangps/workspace/ognn-pytorch/data/ShapeNet/dataset  \
        --generation_dir /home/wangps/workspace/ognn-pytorch/logs/shapenet_eval/test_checkpoint_0327

## Visualize SDFs
python tools/visualize_sdf.py  \
       --filename logs/shapenet_eval/all_${name}/0000.sdf.npy \
       --sdf_scale 0.9


## Extract meshes
python tools/marching_cubes.py  \
       --filelist data/ShapeNet/filelist/all_test.txt  \
       --folder_in logs/shapenet_eval/all_${name}  \
       --folder_out logs/shapenet_eval/all_${name}_mesh/meshes

