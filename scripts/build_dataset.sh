# The mesh is in [-1, 1], and the output point clouds is in [-0.5, 0.5]

## 0. Compute SDFs from meshes

## 1. Sample points from meshes
python tools/sample_pts.py \
       --mesh_folder data/ShapeNet/mesh  \
       --output_folder data/ShapeNet/dataset \
       --filelist data/ShapeNet/filelist/all.txt  \
       --samples 40000  \
       --scale 0.5

## 3. Sample SDFs
python tools/sample_sdf.py \
       --points_folder data/ShapeNet/dataset  \
       --sdf_folder data/ShapeNet/sdf  \
       --output_folder data/ShapeNet/dataset \
       --filelist data/ShapeNet/filelist/all.txt  \
       --sample_num 4  \
       --scale 0.5

## 4. Sample occupancies
python tools/sample_occu.py \
       --sdf_folder data/ShapeNet/sdf  \
       --output_folder data/ShapeNet/dataset \
       --filelist data/ShapeNet/filelist/all.txt  \
       --samples 100000 \
       --scale 0.5
