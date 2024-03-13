# interior room
name=0115_ep600_basic_b8_d8_reg_v1
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 0, \
       SOLVER.resolution 512 \
       DATA.test.filelist data/Shapes/filelist/interior_room.txt \
       DATA.test.location  data/Shapes \
       DATA.test.point_scale 0.8 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.logdir logs/shapes/shapes  \
       SOLVER.alias ${name}_interior_room_512  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1


# all shapes
name=0115_ep600_basic_b8_d8_reg_v1
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 0, \
       SOLVER.resolution 360 \
       DATA.test.filelist data/Shapes/filelist/shapes.txt \
       DATA.test.location  data/Shapes \
       DATA.test.point_scale 0.9 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.logdir logs/shapes/shape  \
       SOLVER.alias ${name}_all_360  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1


# living room
name=0115_ep600_basic_b8_d8_reg_v1
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       SOLVER.gpu 0, \
       SOLVER.resolution 360 \
       DATA.test.filelist data/Shapes/filelist/living_room.txt \
       DATA.test.location  data/Shapes \
       DATA.test.point_scale 0.9 \
       SOLVER.ckpt logs/dfaust/${name}/checkpoints/00600.model.pth \
       SOLVER.logdir logs/shapes/shape  \
       SOLVER.alias ${name}_living_room_360  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1



# fintune on lucy
python completion.py  \
       --config configs/shape.yaml  \
       DATA.train.filelist data/Shapes/filelist/lucy.txt \
       DATA.test.filelist data/Shapes/filelist/lucy.txt \
       SOLVER.alias lucy  \
       SOLVER.gpu 0, \
       SOLVER.max_epoch 4000 \
       SOLVER.ckpt logs/dfaust/0114_ep600_basic_b8_d8_reg_v1/checkpoints/00600.model.pth


# test the finetune of lucy
python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       MODEL.name graph_unet  \
       SOLVER.gpu 1, \
       SOLVER.resolution 360 \
       DATA.test.filelist data/Shapes/filelist/lucy.txt \
       DATA.test.location  data/Shapes \
       DATA.test.point_scale 0.9 \
       SOLVER.ckpt logs/shapes/finetune_lucy_1/checkpoints/01000.model.pth \
       SOLVER.logdir logs/shapes/lucy  \
       SOLVER.alias finetune_360  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1


# fintune on interior room
python completion.py  \
       --config configs/shape.yaml  \
       DATA.test.point_scale 0.6 \
       DATA.train.point_sample_num 1000 \
       DATA.train.filelist data/Shapes/filelist/interior_room.txt \
       DATA.test.point_sample_num 1000 \
       DATA.test.filelist data/Shapes/filelist/interior_room.txt \
       SOLVER.alias interior_room  \
       SOLVER.gpu 1, \
       SOLVER.max_epoch 4000 \
       SOLVER.ckpt logs/dfaust/0114_ep600_basic_b8_d8_reg_v1/checkpoints/00600.model.pth


# fintune on living room
python completion.py  \
       --config configs/shape.yaml  \
       DATA.train.point_sample_num 50000 \
       DATA.train.filelist data/Shapes/filelist/living_room.txt \
       DATA.test.point_sample_num 50000 \
       DATA.test.filelist data/Shapes/filelist/living_room.txt \
       SOLVER.alias living_room  \
       SOLVER.gpu 1, \
       SOLVER.max_epoch 4000 \
       SOLVER.ckpt logs/dfaust/0114_ep600_basic_b8_d8_reg_v1/checkpoints/00600.model.pth


python completion.py  \
       --config configs/dfaust_d8_eval.yaml  \
       SOLVER.gpu 0, \
       SOLVER.resolution 512 \
       DATA.test.filelist data/Shapes/filelist/living_room.txt \
       DATA.test.location  data/Shapes \
       DATA.test.point_scale 0.9 \
       SOLVER.ckpt logs/shapes/finetune_living_room/checkpoints/03000.model.pth \
       SOLVER.logdir logs/shapes/shape  \
       SOLVER.alias finetune_living_room_512  \
       SOLVER.save_sdf True \
       SOLVER.eval_step  -1
