export CUDA_VISIBLE_DEVICES=4
python text_to_image.py   --prompt "a cup of coffee on the table"   --seed 42   --cfg_scale 4.0   --num_images_per_prompt 1   --num_inference_steps 50   --height 1024   --width 1024   --output outputs/coffee.png --model "/share/project/fengyupu/hf_hub/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
