## Train the network
python completion.py  \
       --config configs/shapenet_ae.yaml  



## Generate results for autoencoder
name=0117_ep300_ae
python completion.py  \
      --config configs/shapenet_ae.yaml  \
      DATA.test.batch_size 1 \
      DATA.test.filelist data/ShapeNet/filelist/all_train_im.txt \
      SOLVER.gpu 0,  \
      SOLVER.run evaluate  \
      SOLVER.logdir logs/shapenet_eval/all  \
      SOLVER.ckpt logs/shapenet/${name}/checkpoints/00300.model.pth  \
      SOLVER.alias ${name}_train  \
      SOLVER.resolution 160 \
      SOLVER.sdf_scale 0.9  \
      SOLVER.save_sdf False \
      SOLVER.eval_step -1 

## extract z_code
name=0117_ep300_ae
python autoencoder.py  \
       --config configs/shapenet_ae.yaml  \
       SOLVER.gpu 0,  \
       SOLVER.alias ${name}_zcode  \
       SOLVER.logdir logs/shapenet_eval/all  \
       SOLVER.ckpt logs/shapenet/${name}/checkpoints/00300.model.pth  \
       DATA.train.batch_size 1  \
       DATA.train.shuffle False  \
       SOLVER.run extract_zcode

## forward z_code
python autoencoder.py  \
       --config configs/shapenet_ae.yaml  \
       SOLVER.gpu 1,  \
       SOLVER.alias ${name}_zcode  \
       SOLVER.logdir logs/shapenet_eval/all  \
       SOLVER.ckpt logs/shapenet/${name}/checkpoints/00300.model.pth  \
       SOLVER.sdf_scale 0.9  \
       SOLVER.resolution 160 \
       SOLVER.run decode_zcode


## Rename meshes
python tools/rename_meshes.py  \
       --filelist data/ShapeNet/filelist/all_test_im.txt  \
       --folder_in logs/shapenet_eval/all_${name}  \
       --folder_out logs/shapenet_eval/all_${name}_mesh


## Evaluate meshes
python tools/compute_metrics.py  \
       --mesh_folder logs/shapenet_eval/all_${name}_mesh/meshes  \
       --filename_out logs/shapenet_eval/all_${name}_mesh/metrics.4096.csv \
       --num_samples 4096  \
       --ref_folder data/ShapeNet/mesh.test.gt \
       --filelist data/ShapeNet/filelist/all_test_im.txt

## Visualize SDFs
python tools/visualize_sdf.py  \
       --filename logs/shapenet_eval/all_${name}/0000.sdf.npy \
       --sdf_scale 0.9

