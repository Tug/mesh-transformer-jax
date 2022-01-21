convert:
	python3 to_hf_weights.py --input-ckpt step_150000 --config gpt-j-hf/config.json --output-path out-cedille-150000 --cpu --dtype fp16

convert-abs:
	python3.8 to_hf_weights.py --input-ckpt /mnt/SSD4TB/deep/gpt-j/step_150000 --config /mnt/SSD4TB/deep/gpt-j/mesh-transformer-jax/configs/6B_roto_256.json --output-path /mnt/SSD4TB/deep/gpt-j/out-cedille-150000 --dtype fp16