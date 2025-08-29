# Data
Since the training script uses a dynamic dataset, which generates new training data at the beginning of each epoch, we only provide the dataset generation configuration in `../../data/color/config.json`. The pairwise data used for obtaining NMD is also generated within the scripts. Additionally, the ITI_data `../../data/color/ITI_data.json` is provided as an example for demonstrating Inference-time Intervention (ITI).

# Train
You can train the model with:
```
python train.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --output_path model/default \
  --img_path img/default.png \
  --train_epoch 30 \
  --num_samples 10000
```
* --base_model : Base model path or Hugging Face repo (default: meta-llama/Llama-3.2-3B-Instruct)
*	--output_path  : Path to save the LoRA adapter (default: model/default)
* --img_path  : Path to save the training curve plot (default: img/default.png)
* --train_epoch  : Number of training epochs (default: 30)
* --num_samples  : Amount of training samples (default: 10000)

When training is complete:
* The LoRA adapter will be saved under the folder specified by `--output_path`.
* The best model checkpoint will be located at: `{--output_path}/best/{checkpoint-***}`
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

If the number of NMD samples is insufficient (you’ll need ~5000 samples), rerun the same script with additional test files, e.g. test_1.json, test_2.json, … until you reach the required sample size.

#  Inference-time Intervention (ITI)
In this part, we will utilize NMD last step we obtain to steer the model behavior toward generalization or memorization.