# Data
**Training/Validation data** \
Sample training and validation data is provided under `data/sample`.

If you want to generate data with a different configuration, run:
```
python adder.py \
  --train_size 1000 \
  --test_size 100 \
  --output_path data/default \
  --mem_ratio 0.07
```
* --train_size : Number of training samples (default: 1000)
*	--test_size  : Number of test samples (default: 100)
* --output_path: Directory to save the generated data (default: data/default)
* --mem_ratio  : Ratio of memorized patterns in training data (default: 7%)

Generated data will be saved in the folder specified by `--output_path`.

**Pairwise data** \
*TODO*

# Train
You can train the model using the provided training and validation data with:
```
python train_lora.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --output_path model/default \
  --data_path data/default \
  --img_path img/default.png \
  --train_epoch 50
```
* --base_model : Base model path or Hugging Face repo (default: meta-llama/Llama-3.2-3B-Instruct)
*	--output_path  : Path to save the LoRA adapter (default: model/default)
* --data_path: Directory containing training and validation data (default: data/default)
* --img_path  : Path to save the training curve plot (default: img/default.png)
* --train_epoch  : Number of training epochs (default: 50)

When training is complete:
* The LoRA adapter will be saved under the folder specified by `--output_path`.
* The best model checkpoint will be located at:`{--output_path}/checkpoints/{checkpoint-***}`
* Best model here refers to the one that achieves optimal mem/gen behavior during evaluation.
