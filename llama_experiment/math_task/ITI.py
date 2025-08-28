import argparse
import json
import random
import itertools
import copy
import numpy as np
from scipy.stats import pearsonr
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import PeftModel
from adder import Adder

class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, indices_diff_dict={}, layer_idx=None, alpha=1, applied_layer_idx=tuple(range(26))):
        super().__init__(config)
        self.indices_diff_dict = indices_diff_dict
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.applied_layer_idx = applied_layer_idx

    def forward(
        self, hidden_states, attention_mask=None,
        position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        
        if "position_ids" in kwargs:
            kwargs.pop("position_ids")

        outputs = super().forward(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        
        hidden_states = outputs[0]
        if self.layer_idx in self.applied_layer_idx:
            for idx, shift in self.indices_diff_dict.items():
                hidden_states[:, -1, idx] += self.alpha * shift

        return (hidden_states, ) + outputs[1:]
    
    def set_customized_params(self, indices_diff_dict={}, layer_idx=None, alpha=1, applied_layer_idx=tuple(range(25))):
        self.indices_diff_dict = indices_diff_dict
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.applied_layer_idx = applied_layer_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ITI with LoRA adapter")
    parser.add_argument("--test_data", type=str, default="../../data/math/sample/test.json", help="Path to the JSON file to be loaded")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model path or huggingface repo")
    parser.add_argument( "--adapter", type=str, required=True, help="Adapter model path")
    parser.add_argument( "--device", type=str, default="cuda:0", help="GPU")

    parser.add_argument("--alphas", type=int, nargs="+", default=[1, 3, 5], help="List of alpha values, e.g. --alphas 1 3 5")
    parser.add_argument("--topNs", type=float, nargs="+", default=[0.05, 0.1], help="List of topN fractions, e.g. --topNs 0.05 0.1")

    args = parser.parse_args()
    adder = Adder()

    hyperparameter_combinations = list(itertools.product(args.topNs, args.alphas))
    hyperparam_results = {}
    test_rounds = 1
    max_new_tokens = 300

    with open(args.test_data, "r") as f:
        test_datas = json.load(f)

    ori_results = []

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, args.adapter)

    device = torch.device(args.device)
    model.to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("Running original model inference...")
    for _ in tqdm(range(test_rounds)):
        data = test_datas[_]

        question = data["input"]
        input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]]).to(device)

        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=5,
            do_sample=False,
            early_stopping=True,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        mg_ori = adder.verify_output_type(data, output_text)
        mg_ori = 1 if mg_ori == "mem" else 0 if mg_ori == "gen" else 2
        ori_results.append(mg_ori)

    mg_ori_counter = Counter()
    mg_ori_counter.update(ori_results)
    print("Original result: ", mg_ori_counter)
    print("Original model inference completed!!")

    print("Computing correlations...")
    def calculate_correlation(arrays, labels):
        flattened_arrays = np.array([arr.flatten() for arr in arrays])
        corr_coefficients = np.zeros_like(flattened_arrays[0])

        for i in tqdm(range(flattened_arrays.shape[1])):
            corr_coefficients[i], _ = pearsonr(flattened_arrays[:, i], labels)

        original_shape = arrays[0].shape
        corr_coefficients_2d = corr_coefficients.reshape(original_shape)

        return corr_coefficients_2d
    
    mem_repr_ln2 = np.load(f'../repr_analysis_snapshots/math/mem_repr_hid.npy')
    gen_repr_ln2 = np.load(f'../repr_analysis_snapshots/math/gen_repr_hid.npy')
    print("NMD shape:", mem_repr_ln2.shape, gen_repr_ln2.shape)
    correlations_neuron_wise = calculate_correlation(np.vstack((mem_repr_ln2, gen_repr_ln2)), [1 for _ in range(len(mem_repr_ln2))] + [0 for _ in range(len(gen_repr_ln2))])
    print("Neuron-wise correlations completed!!")

    print("Running ITI...")
    def generate_output(model, input_ids, attention_mask, max_new_tokens, tokenizer):
        torch.cuda.manual_seed_all(42)
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=5, early_stopping=True,
                                attention_mask=attention_mask, eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def process_data(data, ori_result, tokenizer, model_iti_gen, model_iti_mem, model_iti_random, max_new_tokens, device):
        question = data['input']
        input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]]).to(device)
        
        with ThreadPoolExecutor() as executor:
            future_gen = executor.submit(generate_output, model_iti_gen, input_ids, attention_mask, max_new_tokens, tokenizer)
            future_mem = executor.submit(generate_output, model_iti_mem, input_ids, attention_mask, max_new_tokens, tokenizer)
            future_random = executor.submit(generate_output, model_iti_random, input_ids, attention_mask, max_new_tokens, tokenizer)
            
            output_text_iti_gen = future_gen.result()
            output_text_iti_mem = future_mem.result()
            output_text_iti_random = future_random.result()

        mg_gen = adder.verify_output_type(data, output_text_iti_gen)
        mg_mem = adder.verify_output_type(data, output_text_iti_mem)
        mg_random = adder.verify_output_type(data, output_text_iti_random)
        
        mg_gen = 1 if mg_gen == 'mem' else 0 if mg_gen == 'gen' else 2
        mg_mem = 1 if mg_mem == 'mem' else 0 if mg_mem == 'gen' else 2
        mg_random = 1 if mg_random == 'mem' else 0 if mg_random == 'gen' else 2
        
        return ori_result, mg_gen, mg_mem, mg_random

    def run_parallel(test_datas, ori_results, tokenizer, model_iti_gen, model_iti_mem, model_iti_random, max_new_tokens, device, test_rounds):
        results = []
        
        for i in tqdm(range(test_rounds)):
            data = test_datas[i]
            ori_result = ori_results[i]
            result = process_data(data, ori_result, tokenizer, model_iti_gen, model_iti_mem, model_iti_random, max_new_tokens, device)
            results.append(result)
        
        return results

    result_rows = []
    for enu, combination in enumerate(hyperparameter_combinations):
        topN, alpha = combination
        topN = int(topN * gen_repr_ln2.shape[1] * gen_repr_ln2.shape[2])

        flat_array = np.absolute(correlations_neuron_wise.flatten())
        sorted_indices = np.argsort(flat_array)[::-1][:topN]
        sorted_indices_2d = np.unravel_index(sorted_indices, correlations_neuron_wise.shape)
        sorted_indices_2d = np.stack(sorted_indices_2d, axis=-1)
        pairwise_diff_ln2_mean = (np.array(mem_repr_ln2).mean(axis=0) - np.array(gen_repr_ln2).mean(axis=0))

        sorted_indices_diff_ln2_by_layer = defaultdict(list)
        sorted_indices_diff_ln2_by_layer_mem = defaultdict(list)
        sorted_indices_diff_ln2_by_layer_gen = defaultdict(list)
        sorted_indices_diff_ln2_by_layer_random = defaultdict(list)
        sorted_indices_mem_ln2_by_layer = defaultdict(list)
        sorted_indices_gen_ln2_by_layer = defaultdict(list)
        sorted_indices_mem_ln2_by_layer_shap = defaultdict(list)
        sorted_indices_gen_ln2_by_layer_shap = defaultdict(list)

        max_abs_NMD = 0
        for i, j in sorted_indices_2d:
            max_abs_NMD = max(max_abs_NMD, abs(pairwise_diff_ln2_mean[i, j]))
            sorted_indices_diff_ln2_by_layer[i].append((j, pairwise_diff_ln2_mean[i, j]))
            sorted_indices_diff_ln2_by_layer_mem[i].append((j, pairwise_diff_ln2_mean[i, j]))
            sorted_indices_diff_ln2_by_layer_gen[i].append((j, -pairwise_diff_ln2_mean[i, j]))
            sorted_indices_mem_ln2_by_layer[i].append((j, mem_repr_ln2[:, i, j].mean()))
            sorted_indices_gen_ln2_by_layer[i].append((j, gen_repr_ln2[:, i, j].mean()))
        
        # randomly do ITI
        rows = mem_repr_ln2[0].shape[0]
        cols = mem_repr_ln2[0].shape[1]
        all_coords = [(i, j) for i in range(rows) for j in range(cols)]
        random_items = random.sample(all_coords, topN)

        for (i, j) in random_items:
            v = random.uniform(-max_abs_NMD, max_abs_NMD)
            sorted_indices_diff_ln2_by_layer_random[i].append((j, v))


        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model_iti_gen = LlamaForCausalLM.from_pretrained(args.base_model)
        model_iti_gen = PeftModel.from_pretrained(model_iti_gen, args.adapter)
        model_iti_mem = LlamaForCausalLM.from_pretrained(args.base_model)
        model_iti_mem = PeftModel.from_pretrained(model_iti_mem, args.adapter)
        model_iti_random = LlamaForCausalLM.from_pretrained(args.base_model)
        model_iti_random = PeftModel.from_pretrained(model_iti_random, args.adapter)
            
        for i in range(len(model_iti_gen.model.model.layers)):
            custom_block = copy.deepcopy(model_iti_gen.model.model.layers[i])
            custom_block.__class__ = CustomLlamaDecoderLayer
            custom_block.set_customized_params({k: v for k, v in sorted_indices_diff_ln2_by_layer_gen[i + 1][:topN]}, i, alpha)
            model_iti_gen.model.model.layers[i] = custom_block
            
            custom_block = copy.deepcopy(model_iti_mem.model.model.layers[i])
            custom_block.__class__ = CustomLlamaDecoderLayer
            custom_block.set_customized_params({k: v for k, v in sorted_indices_diff_ln2_by_layer_mem[i + 1][:topN]}, i, alpha)
            model_iti_mem.model.model.layers[i] = custom_block

            custom_block = copy.deepcopy(model_iti_random.model.model.layers[i])
            custom_block.__class__ = CustomLlamaDecoderLayer
            custom_block.set_customized_params({k: v for k, v in sorted_indices_diff_ln2_by_layer_random[i + 1][:topN]}, i, alpha)
            model_iti_random.model.model.layers[i] = custom_block

        model_iti_gen = model_iti_gen.to(device)
        model_iti_mem = model_iti_mem.to(device)
        model_iti_random = model_iti_random.to(device)

        results = run_parallel(test_datas, ori_results, tokenizer, model_iti_gen, model_iti_mem, model_iti_random, max_new_tokens, device, test_rounds)
        c = Counter()
        c.update(results)
        
        hyperparam_results[combination] = c
        result_row = [combination, 
            sum([c[k] for k in c if k[0] == 0 and k[1] == 0])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 0 and k[3] == 0])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 1 and k[2] == 1])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[3] == 1])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[1] == 0])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[1] == 1])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[1] == 2])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[3] == 0])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[3] == 1])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 1 and k[3] == 2])/ sum([c[k] for k in c if k[0] == 1]),
            sum([c[k] for k in c if k[0] == 0 and k[2] == 0])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 0 and k[2] == 1])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 0 and k[2] == 2])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 0 and k[3] == 0])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 0 and k[3] == 1])/ sum([c[k] for k in c if k[0] == 0]),
            sum([c[k] for k in c if k[0] == 0 and k[3] == 2])/ sum([c[k] for k in c if k[0] == 0]),
            ]
        result_rows.append(result_row)
        print(combination)
        
        print(f"original: gen: {sum([c[k] for k in c if k[0] == 0]) / test_rounds:.4f}, mem: {sum([c[k] for k in c if k[0] == 1]) / test_rounds:.4f}, other: {sum([c[k] for k in c if k[0] == 2]) / test_rounds:.4f}")
        print(f"gen: gen: {sum([c[k] for k in c if k[1] == 0]) / test_rounds:.4f}, mem: {sum([c[k] for k in c if k[1] == 1]) / test_rounds:.4f}, other: {sum([c[k] for k in c if k[1] == 2]) / test_rounds:.4f}")
        print(f"mem: gen: {sum([c[k] for k in c if k[2] == 0]) / test_rounds:.4f}, mem: {sum([c[k] for k in c if k[2] == 1]) / test_rounds:.4f}, other: {sum([c[k] for k in c if k[2] == 2]) / test_rounds:.4f}")
        print(f"random: gen: {sum([c[k] for k in c if k[3] == 0]) / test_rounds:.4f}, mem: {sum([c[k] for k in c if k[3] == 1]) / test_rounds:.4f}, other: {sum([c[k] for k in c if k[3] == 2]) / test_rounds:.4f}")
        
        print(f'gen model:')
        print(f'  original gen:')
        print(f'    gen  : {sum([c[k] for k in c if k[0] == 0 and k[1] == 0])} ({sum([c[k] for k in c if k[0] == 0 and k[1] == 0])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'    mem  : {sum([c[k] for k in c if k[0] == 0 and k[1] == 1])} ({sum([c[k] for k in c if k[0] == 0 and k[1] == 1])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'    other: {sum([c[k] for k in c if k[0] == 0 and k[1] == 2])} ({sum([c[k] for k in c if k[0] == 0 and k[1] == 2])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'  original mem:')
        print(f'    gen  : {sum([c[k] for k in c if k[0] == 1 and k[1] == 0])} ({sum([c[k] for k in c if k[0] == 1 and k[1] == 0])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'    mem  : {sum([c[k] for k in c if k[0] == 1 and k[1] == 1])} ({sum([c[k] for k in c if k[0] == 1 and k[1] == 1])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'    other: {sum([c[k] for k in c if k[0] == 1 and k[1] == 2])} ({sum([c[k] for k in c if k[0] == 1 and k[1] == 2])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'mem model:')
        print(f'  original gen:')
        print(f'    gen  : {sum([c[k] for k in c if k[0] == 0 and k[2] == 0])} ({sum([c[k] for k in c if k[0] == 0 and k[2] == 0])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'    mem  : {sum([c[k] for k in c if k[0] == 0 and k[2] == 1])} ({sum([c[k] for k in c if k[0] == 0 and k[2] == 1])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'    other: {sum([c[k] for k in c if k[0] == 0 and k[2] == 2])} ({sum([c[k] for k in c if k[0] == 0 and k[2] == 2])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'  original mem:')
        print(f'    gen  : {sum([c[k] for k in c if k[0] == 1 and k[2] == 0])} ({sum([c[k] for k in c if k[0] == 1 and k[2] == 0])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'    mem  : {sum([c[k] for k in c if k[0] == 1 and k[2] == 1])} ({sum([c[k] for k in c if k[0] == 1 and k[2] == 1])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'    other: {sum([c[k] for k in c if k[0] == 1 and k[2] == 2])} ({sum([c[k] for k in c if k[0] == 1 and k[2] == 2])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'random model:')
        print(f'  original gen:')
        print(f'    gen  : {sum([c[k] for k in c if k[0] == 0 and k[3] == 0])} ({sum([c[k] for k in c if k[0] == 0 and k[3] == 0])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'    mem  : {sum([c[k] for k in c if k[0] == 0 and k[3] == 1])} ({sum([c[k] for k in c if k[0] == 0 and k[3] == 1])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'    other: {sum([c[k] for k in c if k[0] == 0 and k[3] == 2])} ({sum([c[k] for k in c if k[0] == 0 and k[3] == 2])/ sum([c[k] for k in c if k[0] == 0]):.4f}%)')
        print(f'  original mem:')
        print(f'    gen  : {sum([c[k] for k in c if k[0] == 1 and k[3] == 0])} ({sum([c[k] for k in c if k[0] == 1 and k[3] == 0])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'    mem  : {sum([c[k] for k in c if k[0] == 1 and k[3] == 1])} ({sum([c[k] for k in c if k[0] == 1 and k[3] == 1])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')
        print(f'    other: {sum([c[k] for k in c if k[0] == 1 and k[3] == 2])} ({sum([c[k] for k in c if k[0] == 1 and k[3] == 2])/ sum([c[k] for k in c if k[0] == 1]):.4f}%)')