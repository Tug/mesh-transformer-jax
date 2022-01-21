convert:
	python3 to_hf_weights.py --input-ckpt step_150000 --config configs/6B_roto_256.json --output-path out-cedille-150000 --dtype fp16
