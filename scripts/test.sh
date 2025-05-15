cd ..
CUDA_VISIBLE_DEVICES=$1 python solve.py \
    --base_model flux \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory \
    --workdir workdir/test