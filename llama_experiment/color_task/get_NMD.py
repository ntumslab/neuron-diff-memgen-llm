import os
import numpy as np
import argparse
import gc
import random
import torch
import json
import string
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
from peft import PeftModel
from pathlib import Path
from baukit import Trace, TraceDict
from itertools import islice

global config_file
global names
global roles
with open('../../data/color/config.json.json', "r") as f:
    config_file = json.load(f)
names = list(config_file["name_color_map"].keys())
roles = config_file["roles"]  
train_name_colors = config_file['name_color_map']

def main():

    parser = argparse.ArgumentParser(description="Read and process a JSON file.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of pairwise data for obtaining NMD")
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

    def verify_output_type(output, question, answer, train_name_colors):
        # Find the position of the first occurrence of "?"
        question_mark_index = output.find("?")
        # question_mark_index = output.find("?", question_mark_index + 1)
        substring = output[question_mark_index+1:]
        if len(substring.split()) == 0 or len(substring.split()[0].split('.')) == 0:
            return 2, ""

        color = substring.split()[0].split('.')[0].strip(string.punctuation)
        # Labels == 0 -> gen, labels == 1 -> mem
        if color == answer:
            label = 0  # generalization
        elif color in train_name_colors[question]:
            label = 1  # memorization
        else:
            label = 2 # error
        return label, color

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
    
        concat_and_save(f'../repr_analysis_snapshots/color/{name}.npy', arr)
        arr.clear()
    
    def clause_to_prompt(sentences):
        return ' '.join(sentences)

    def generate_test_data(return_sentences=False):
        all_colors = config_file["all_colors"] 
        
        num_names = random.randint(4,8)
        num_roles = 5
        selected_names = random.sample(names, num_names)
        selected_roles = random.sample(roles, num_roles)
        
        target = selected_names[0]
        target_role = selected_roles[0]
        
        colors = random.sample(list(set(all_colors) - set(config_file["name_color_map"][target])), num_roles)
        while colors[0] in config_file["name_color_map"][target]:
            colors = random.sample(config_file["all_colors"], num_roles)
        target_color = colors[0]

        clue = selected_names[1]
        clue_role = target_role

        sentences = []
        role_color_map = {} 
        role_color_map[selected_roles[0]] = target_color
        
        for i in range(1,num_roles):
            role_color_map[selected_roles[i]] = colors[i]

        for i in range(2, num_names):
            role = random.choice(selected_roles)
            sentences.append(f"{selected_names[i]} is {role}.")
            if role != target_role:
                sentences.append(f"{selected_names[i]} is {role_color_map[role]}.")

        random.shuffle(sentences)
        sentences = sentences[:7]
        sentences.insert(0, f"{clue} is {clue_role}.")
        sentences.insert(0, f"{clue} is {role_color_map[clue_role]}.")
        sentences.insert(0, f"{target} is {target_role}.")
        random.shuffle(sentences)
        sentences.append(f"what color is {target}?")
        
        if return_sentences:
            return ' '.join(sentences) + ' ' + target_color, sentences, target_color
        return ' '.join(sentences) + ' ' + target_color

    def evaluate_pairwise(model, tokenizer, device, num_samples=10000):
        gen = 0
        mem = 0
        err = 0
        diff = 0

        mem_repr_hid, gen_repr_hid = [], []
        mem_repr_qbase, gen_repr_qbase = [], []
        mem_repr_qlora, gen_repr_qlora = [], []
        mem_repr_vbase, gen_repr_vbase = [], []
        mem_repr_vlora, gen_repr_vlora = [], []

        for i in tqdm(range(num_samples)):

            test_datum, test_sentences, answer = generate_test_data(return_sentences=True)
            prompt = test_datum.split('? ')[0] + '?'
            question = test_sentences[-1].split()[-1][:-1]
            prompt2 = clause_to_prompt(random.sample(test_sentences[:-1], len(test_sentences) - 1) + test_sentences[-1:])

            input_ids_1 = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask_1 = torch.tensor([[1] * input_ids_1.shape[1]]).to(device)
            output = model.generate(input_ids_1, max_length=len(input_ids_1[0]) + 2, num_beams=5, 
                                    early_stopping=True, do_sample=False,
                                    attention_mask=attention_mask_1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            label1, _ = verify_output_type(output_text, question, answer, train_name_colors)
   
            input_ids_2 = tokenizer.encode(prompt2, return_tensors="pt").to(device)
            attention_mask_2 = torch.tensor([[1] * input_ids_2.shape[1]]).to(device)
            output = model.generate(input_ids_2, max_length=len(input_ids_2[0]) + 2, num_beams=5, 
                                    early_stopping=True, do_sample=False,
                                    attention_mask=attention_mask_2, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            label2, _ = verify_output_type(output_text, question, answer, train_name_colors)

            if label1 == 2 or label2 == 2:
                continue
            elif label1 != label2: # Different Occur!
                diff += 1
                hidden_states1, q_base1, q_lora1, v_base1, v_lora1 = get_activations_bau(model, input_ids_1, attention_mask_1)
                hidden_states2, q_base2, q_lora2, v_base2, v_lora2 = get_activations_bau(model, input_ids_2, attention_mask_2)

                if label1 == 0 and label2 == 1:
                    mem_repr_hid.append(hidden_states2)
                    mem_repr_qbase.append(q_base2), mem_repr_qlora.append(q_lora2)
                    mem_repr_vbase.append(v_base2), mem_repr_vlora.append(v_lora2)
                    gen_repr_hid.append(hidden_states1)
                    gen_repr_qbase.append(q_base1), gen_repr_qlora.append(q_lora1)
                    gen_repr_vbase.append(v_base1), gen_repr_vlora.append(v_lora1)
                elif label1 == 1 and label2 == 0:
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
        print(f"Gen: {gen / total}, Mem: {mem / total}, Diff: {diff / total / 2}")

    evaluate_pairwise(model, tokenizer, device, args.num_samples)

if __name__ == "__main__":
    main()