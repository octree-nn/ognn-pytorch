## Train
alias=ref_0401
python main.py  \
       --config configs/shapenet.yaml  \
       SOLVER.gpu 0,1,2,3  \
       SOLVER.port 10001   \
       SOLVER.alias ${alias}

## Test
python main.py  \
       SOLVER.ckpt logs/shapenet/shapenet_${alias}/checkpoints/00300.model.pth \
       SLOVER.gpu 0,  \
       SOLVER.alias ${alias}


## Evaluate meshes
cd ../ConvONet
python eval_meshes.py  \
      configs/pointcloud/shapenet.yaml  \
      --dataset_folder ../ognn-pytorch/data/ShapeNet/dataset  \
      --generation_dir ../ognn-pytorch/logs/shapenet_eval/test_${alias}


## Visualize SDFs
python tools/visualize_sdf.py  \
       --filename logs/shapenet_eval/all_${name}/0000.sdf.npy \
       --sdf_scale 0.9

