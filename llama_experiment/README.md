# Data
**Training/Validation data** \
Sample training and validation data is provided under `data/sample`.

If you want to generate data with a different configuration, run:
```
python adder.py \
  --train_size 1000 \
  --test_size 100 \
  --output_path data/sample \
  --mem_ratio 0.07
```
* --train_size : Number of training samples (default: 1000)
*	--test_size  : Number of test samples (default: 100)
* --output_path: Directory to save the generated data (default: data/sample)
* --mem_ratio  : Ratio of memorized patterns in training data (default: 7%)

Generated data will be saved in the folder specified by `--output_path`.

**Pairwise data** \

# Train
