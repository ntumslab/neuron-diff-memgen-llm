from transformers import AutoTokenizer, LlamaForCausalLM
from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm
from peft import PeftModel
from baukit import Trace, TraceDict
from pathlib import Path
import os
import gc
from itertools import islice
import numpy as np
import argparse


def main():

    parser = argparse.ArgumentParser(description="Read and process a JSON file.")
    parser.add_argument("--test_file", type=str, default="../../data/math/sample_pairwise/test_0.json", help="Path to the JSON file to be loaded.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model path or huggingface repo")
    parser.add_argument("--adapter_path", type=str, required=True, help="Adapter model path")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU")
    args = parser.parse_args()

    device = args.device

    print(args.adapter_path, device)
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load test data
    def preprocess_data(data):
        tokenizer.padding_side = 'left'
        encodings = tokenizer(
            [item['input'] for item in data],
            truncation=True,
            padding="max_length",
            max_length=300,
            return_tensors="pt"
        )
        labels = tokenizer(
            [item['output'] for item in data],
            truncation=True,
            padding="max_length",
            max_length=300,
            return_tensors="pt"
        )['input_ids']

        mem_patterns = [item['mem_pattern_hashed'] for item in data]
        return encodings, labels, mem_patterns

    class TextGenerationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels, mem_patterns):
            self.encodings = encodings
            self.labels = labels
            self.mem_patterns = mem_patterns

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            if self.labels != None:
                item['labels'] = self.labels[idx].clone().detach()
            if self.mem_patterns != None:
                item['mem_pattern_hashed'] = self.mem_patterns[idx]  # Add mem_pattern_hashed
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])
    
    
    with open(args.test_file, 'r') as file:
        test_data = json.load(file)

    first_test_data = [group[0] for group in test_data]
    second_test_data = [group[1] for group in test_data]
    eval_1_encodings, eval_1_labels, eval_1_mem_patterns = preprocess_data(first_test_data)
    eval_2_encodings, eval_2_labels, eval_2_mem_patterns = preprocess_data(second_test_data)
    print(args.test_file)
    print("Create evaluation data...")
    eval_1_dataset = TextGenerationDataset(eval_1_encodings, eval_1_labels, eval_1_mem_patterns)
    eval_2_dataset = TextGenerationDataset(eval_2_encodings, eval_2_labels, eval_2_mem_patterns)

    def verify_output_type(data, text):
        text = text[len(data['input']):]

        mem_pattern_hashed = data['mem_pattern_hashed']
        if text.startswith(f'<{mem_pattern_hashed}>'):
            return 'mem'
        
        lines = text.split('\n')
        if '<scratch>' not in lines or '</scratch>' not in lines:
            return 'unknown'
        
        ans = lines[lines.index('</scratch>') + 1]
        gt_lines = data['output'].split('\n')
        gt = gt_lines[gt_lines.index('</scratch>') + 1]

        if gt == ans:
            return 'gen'
        
        return 'unknown'

    def get_activations_bau(model, input_ids, attention_mask): 
        Q_BASE = [f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer" for i in range(model.config.num_hidden_layers)]
        Q_LORA = [f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.default" for i in range(model.config.num_hidden_layers)]
        V_BASE = [f"base_model.model.model.layers.{i}.self_attn.v_proj.base_layer" for i in range(model.config.num_hidden_layers)]
        V_LORA = [f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default" for i in range(model.config.num_hidden_layers)]

        with torch.no_grad():
            with TraceDict(model, Q_BASE+Q_LORA+V_BASE+V_LORA , retain_input=True) as ret:
                output = model(input_ids, output_hidden_states = True, attention_mask=attention_mask)

                hidden_states = output.hidden_states
                hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
                hidden_states = hidden_states.detach().cpu().numpy()

                q_base = [ret[qb].output.squeeze().detach().cpu() for qb in Q_BASE]
                q_base = torch.stack(q_base, dim = 0).squeeze().numpy()
                q_lora = [ret[qb].output.squeeze().detach().cpu() for qb in Q_LORA]
                q_lora = torch.stack(q_lora, dim = 0).squeeze().numpy()
                
                v_base = [ret[qb].output.squeeze().detach().cpu() for qb in V_BASE]
                v_base = torch.stack(v_base, dim = 0).squeeze().numpy()
                v_lora = [ret[qb].output.squeeze().detach().cpu() for qb in V_LORA]
                v_lora = torch.stack(v_lora, dim = 0).squeeze().numpy()

        return hidden_states[:,-1,:], q_base[:,-1,:], q_lora[:,-1,:], v_base[:,-1,:], v_lora[:,-1,:]

    def save_and_clear(name, arr):
        def concat_and_save(path, new_data):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            if os.path.exists(path):
                old_data = np.load(path, allow_pickle=True)
                combined = np.concatenate([old_data, np.array(new_data)], axis=0)
            else:
                combined = np.array(new_data)
            if name == 'gen_repr_vlora':
                print(f"Save: {len(combined)} sample.")
            np.save(path, combined)
            
            del combined
            gc.collect()
    
        concat_and_save(f'../repr_analysis_snapshots/math/{name}.npy', arr)
        arr.clear()

    def evaluate_pairwise(model, tokenizer, eval_1_dataset, eval_2_dataset, device, batch_size, max_new_tokens=300):
        eval_1_loader = DataLoader(eval_1_dataset, batch_size=batch_size)
        eval_2_loader = DataLoader(eval_2_dataset, batch_size=batch_size)

        gen = 0
        mem = 0
        err = 0
        diff = 0

        mem_repr_hid, gen_repr_hid = [], []
        mem_repr_qbase, gen_repr_qbase = [], []
        mem_repr_qlora, gen_repr_qlora = [], []
        mem_repr_vbase, gen_repr_vbase = [], []
        mem_repr_vlora, gen_repr_vlora = [], []

        for batch_1, batch_2 in tqdm(islice(zip(eval_1_loader, eval_2_loader), None),
                                        total=min(len(eval_1_loader), len(eval_2_loader))):
            with torch.no_grad():
                input_ids_1 = batch_1["input_ids"].to(device)
                attention_mask_1 = batch_1["attention_mask"].to(device)
                generated_ids_1 = model.generate(
                    input_ids=input_ids_1,
                    attention_mask=attention_mask_1,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
                
                generated_texts_1 = tokenizer.batch_decode(generated_ids_1, skip_special_tokens=True)
                input_texts_1 = tokenizer.batch_decode(input_ids_1, skip_special_tokens=True)
                output_texts_1 = tokenizer.batch_decode(batch_1["labels"], skip_special_tokens=True)
                
                result1_type = []
                for i in range(len(generated_texts_1)):
                    data_1 = {
                        'input': input_texts_1[i],
                        'output': output_texts_1[i],
                        'mem_pattern_hashed': batch_1['mem_pattern_hashed'][i]
                    }
                    result_1_type = verify_output_type(data_1, generated_texts_1[i])
                    if result_1_type == 'unknown':
                        err += 1
                    elif result_1_type == 'gen':
                        gen += 1
                    elif result_1_type == 'mem':
                        mem += 1
                    result1_type.append(result_1_type)
                
                input_ids_2 = batch_2["input_ids"].to(device)
                attention_mask_2 = batch_2["attention_mask"].to(device)
                
                generated_ids_2 = model.generate(
                    input_ids=input_ids_2,
                    attention_mask=attention_mask_2,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )
                
                generated_texts_2 = tokenizer.batch_decode(generated_ids_2, skip_special_tokens=True)
                input_texts_2 = tokenizer.batch_decode(input_ids_2, skip_special_tokens=True)
                output_texts_2 = tokenizer.batch_decode(batch_2["labels"], skip_special_tokens=True)
                
                result2_type = []
                for i in range(len(generated_texts_2)):
                    data_2 = {
                        'input': input_texts_2[i],
                        'output': output_texts_2[i],
                        'mem_pattern_hashed': batch_2['mem_pattern_hashed'][i]
                    }
                    result_2_type = verify_output_type(data_2, generated_texts_2[i])
                    if result_2_type == 'unknown':
                        err += 1
                    elif result_2_type == 'gen':
                        gen += 1
                    elif result_2_type == 'mem':
                        mem += 1
                    result2_type.append(result_2_type)

            # Get representation
            for i, (type1, type2) in enumerate(zip(result1_type, result2_type)):
                if type1 == 'unknown' or type2 == 'unknown':
                    continue
                elif type1 != type2: # Different Occur!
                    diff += 1
                    hidden_states1, q_base1, q_lora1, v_base1, v_lora1 = get_activations_bau(model, input_ids_1[i].unsqueeze(0), attention_mask_1[i].unsqueeze(0))
                    hidden_states2, q_base2, q_lora2, v_base2, v_lora2 = get_activations_bau(model, input_ids_2[i].unsqueeze(0), attention_mask_2[i].unsqueeze(0))

                    if type1 == 'gen' and type2 == 'mem':
                        mem_repr_hid.append(hidden_states2)
                        mem_repr_qbase.append(q_base2), mem_repr_qlora.append(q_lora2)
                        mem_repr_vbase.append(v_base2), mem_repr_vlora.append(v_lora2)
                        gen_repr_hid.append(hidden_states1)
                        gen_repr_qbase.append(q_base1), gen_repr_qlora.append(q_lora1)
                        gen_repr_vbase.append(v_base1), gen_repr_vlora.append(v_lora1)
                    elif type1 == 'mem' and type2 == 'gen':
                        mem_repr_hid.append(hidden_states1)
                        mem_repr_qbase.append(q_base1), mem_repr_qlora.append(q_lora1)
                        mem_repr_vbase.append(v_base1), mem_repr_vlora.append(v_lora1)
                        gen_repr_hid.append(hidden_states2)
                        gen_repr_qbase.append(q_base2), gen_repr_qlora.append(q_lora2)
                        gen_repr_vbase.append(v_base2), gen_repr_vlora.append(v_lora2)
            
            if len(mem_repr_hid) >= 100:
                save_and_clear('mem_repr_hid', mem_repr_hid), save_and_clear('gen_repr_hid', gen_repr_hid)
                save_and_clear('mem_repr_qbase', mem_repr_qbase), save_and_clear('gen_repr_qbase', gen_repr_qbase)
                save_and_clear('mem_repr_qlora', mem_repr_qlora), save_and_clear('gen_repr_qlora', gen_repr_qlora)
                save_and_clear('mem_repr_vbase', mem_repr_vbase), save_and_clear('gen_repr_vbase', gen_repr_vbase)
                save_and_clear('mem_repr_vlora', mem_repr_vlora), save_and_clear('gen_repr_vlora', gen_repr_vlora)
        
        if len(mem_repr_hid) > 0:
            save_and_clear('mem_repr_hid', mem_repr_hid), save_and_clear('gen_repr_hid', gen_repr_hid)
            save_and_clear('mem_repr_qbase', mem_repr_qbase), save_and_clear('gen_repr_qbase', gen_repr_qbase)
            save_and_clear('mem_repr_qlora', mem_repr_qlora), save_and_clear('gen_repr_qlora', gen_repr_qlora)
            save_and_clear('mem_repr_vbase', mem_repr_vbase), save_and_clear('gen_repr_vbase', gen_repr_vbase)
            save_and_clear('mem_repr_vlora', mem_repr_vlora), save_and_clear('gen_repr_vlora', gen_repr_vlora)

        total = gen + mem + err
        print(f"Gen: {gen / total}, Mem: {mem / total}, Diff: {diff / total}")

    evaluate_pairwise(model, tokenizer, eval_1_dataset, eval_2_dataset, device, 8)

if __name__ == "__main__":
    main()