# Data
**Training/Validation data** \
Sample training and validation data is provided under `../../data/cipher/sample`.

If you want to generate data with a different configuration, run:
```
python ciphertext.py \
  --train_size 1000 \
  --test_size 100 \
  --output_path ../../data/cipher/default \
  --mem_ratio 0.07 \
  --type train
```
* --train_size : Number of training samples (default: 10000)
*	--test_size  : Number of test samples (default: 100)
* --output_path: Directory to save the generated data (default: ../../data/cipher/default)
* --mem_ratio  : Ratio of memorized patterns in training data (default: 7%)
* --type: Type of generated data (default: train)

Generated data will be saved in the folder specified by `--output_path`.

**Pairwise data** \
Sample pairwise data is provided under `../../data/cipher/sample_pairwise`.

If you want to generate pairwise dataset, run:
```
python ciphertext.py \
  --output_path ../../data/cipher/default_pairwise \
  --type pairwise
```
* --output_path: Directory to save the generated data (default: ../../data/cipher/default)
* --type: Type of generated data (default: train)

# Train
You can train the model using the provided training and validation data with:
```
python train_lora.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --output_path model/default \
  --data_path ../../data/math/default \
  --img_path img/default.png \
  --train_epoch 50
```
* --base_model : Base model path or Hugging Face repo (default: meta-llama/Llama-3.2-3B-Instruct)
*	--output_path  : Path to save the LoRA adapter (default: model/default)
* --data_path: Directory containing training and validation data (default: ../../data/math/default)
* --img_path  : Path to save the training curve plot (default: img/default.png)
* --train_epoch  : Number of training epochs (default: 50)

When training is complete:
* The LoRA adapter will be saved under the folder specified by `--output_path`.
* The best model checkpoint will be located at:`{--output_path}/checkpoints/{checkpoint-***}`
* Best model here refers to the one that achieves optimal mem/gen behavior during evaluation.

# Compute Neuron Mean Differentiation (NMD)
After training the model, you can compute **Neuron Mean Differentiation (NMD)** by running inference on a pairwise dataset:
```
python get_NMD.py \
  --test_file ../../data/math/sample_pairwise/test_0.json
  --base_model meta-llama/Llama-3.2-3B-Instruct
  --adapter_path {path/to/your_best_adapter}
  --device cuda:0
```
* --test_file: Path to the pairwise data (default: ../../data/math/sample_pairwise/test_0.json)
* --base_model: Base model path or Hugging Face repo (default: meta-llama/Llama-3.2-3B-Instruct)
* --adapter_path: Path to your best adapter model **(required)**
* --device: GPU device (default: cuda:0)

When the run completes, the NMD files will be saved under `../repr_analysis_snapshots/math`, including:
* gen_repr_hid, gen_repr_qbase, gen_repr_qlora, gen_repr_vbase, gen_repr_vlora
* mem_repr_hid, mem_repr_qbase, mem_repr_qlora, mem_repr_vbase, mem_repr_vlora

If the number of NMD samples is insufficient (you’ll need ~6000 samples), rerun the same script with additional test files, e.g. test_1.json, test_2.json, … until you reach the required sample size.

#  Inference-time Intervention (ITI)
In this part, we will utilize NMD last step we obtain to steer the model behavior toward generalization or memorization.