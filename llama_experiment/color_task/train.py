import datasets
import ipdb
import argparse
import os
import random
import json
import torch
import string
import matplotlib.pyplot as plt
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType


global config_file
global names
global roles

with open('../../data/color/config.json', "r") as f:
    config_file = json.load(f)
names = list(config_file["name_color_map"].keys())
roles = config_file["roles"]  
train_name_colors = config_file['name_color_map']

def generate_data(return_sentences=False):
    num_names = random.randint(4,8)
    num_roles = 5
    selected_names = random.sample(names, num_names)
    selected_roles = random.sample(roles, num_roles)
    target = selected_names[0]
    target_role = selected_roles[0]
    
    colors = random.sample(config_file["all_colors"], num_roles)
    while colors[0] not in config_file["name_color_map"][target]:
        colors = random.sample(config_file["all_colors"], num_roles)

    clue = selected_names[1]
    clue_role = target_role

    sentences = []
    role_color_map = {} 
    for i in range(num_roles):
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
        return ' '.join(sentences) + ' ' + colors[0], sentences, colors[0]
    return ' '.join(sentences) + ' ' + colors[0]
        
def generate_signal_data(return_sentences=False):
    target = random.sample(names, 1)[0]
    color = random.sample(config_file["name_color_map"][target], 1)[0]
    sentence = f"what color is {target}?"

    if return_sentences:
        return sentence + ' ' + color, sentence, color
    return sentence + ' ' + color

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
    
    for i in range(1, num_roles):
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


class DynamicDataset(Dataset):
    def __init__(self, tokenizer, num_samples, max_length=70, threshold=1):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.threshold = threshold
        # static version
        self.static_sentences = [generate_data() for _ in range(int(num_samples * threshold))]
        self.signal_sentences = [generate_signal_data() for _ in range(int(num_samples * (1 - threshold)))]
        self.static_sentences.extend(self.signal_sentences)
        random.shuffle(self.static_sentences)
        self.static_tokens = [self.tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length) for s in self.static_sentences]
        self.static_input_ids = [t['input_ids'].squeeze() for t in self.static_tokens]
        self.static_attention_masks = [t['attention_mask'].squeeze() for t in self.static_tokens]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if random.random() < self.threshold:
            sentence = generate_signal_data()
        else:
            sentence = generate_data()
        sentence = generate_data()
        tokens = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
        }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = []
        self.correlation = []
        self.best_metric = float('-inf')
        self.best_gen = float('-inf')
        self.num_saved_models = 0
        self.max_saved_models = 5 

    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        model = self.model
        tokenizer = self.tokenizer
        
        model.eval()
        tokenizer.padding_side = 'left'

        gen = 0
        mem = 0
        err = 0

        labels = []
        eval_num = len(eval_dataset)

        all_outputs = []
        batch_size = 32
        num_batches = eval_num // batch_size + (1 if eval_num % batch_size != 0 else 0)
        
        # Process each batch prediction
        for i in tqdm(range(num_batches)):
            # Prepare the batch inputs and attention masks
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, eval_num)
            input_texts = [data for data in eval_dataset[batch_start:batch_end]['input']]
            inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=70).input_ids.to(self.args.device)
            attention_mask = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=70).attention_mask.to(self.args.device)

            # Generate text for the batch
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=70,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )

            # Decode the generated IDs to strings
            batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Append the decoded outputs to the final list
            all_outputs.extend(batch_outputs)

        # Batch evaluation
        for i, output in enumerate(all_outputs):
            # Find the position of the first occurrence of "?"
            question_mark_index = output.find("?")
            substring = output[question_mark_index+1:]
            try:
                generated_answer = substring.split()[0].split('.')[0].strip(string.punctuation)
            except:
                err += 1
                continue

            # Labels == 0 -> gen, labels == 1 -> mem
            if generated_answer == eval_dataset[i]['output']:
                gen += 1
                labels.append(0)
            elif generated_answer in config_file['name_color_map'][eval_dataset[i]['input'].split()[-1][:-1]]:
                mem += 1
                labels.append(1)
            else:
                err += 1             

        model.train()
        tokenizer.padding_side = 'right'

        total = gen+mem+err
        print(f"Gen: {gen/total}, Mem: {mem/total}")
        self.result.append([gen/total, mem/total])

        with open('result.txt', 'a') as f:
            f.write(f"{[gen/total, mem/total]}\n")
        
        # Check if current model performs better than the previous one
        current_metric = gen*mem
        current_gen = gen
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self._save_custom_checkpoint()
        elif current_gen > self.best_gen:
            self.best_gen = gen
            self._save_custom_checkpoint()

        return {"eval_loss": (gen+mem)/total, "gen": gen/total, "mem": mem/total}

    def get_mg_record(self):
        return self.result

    def _save_custom_checkpoint(self):
        # Create a directory for checkpoints if it doesn't exist
        checkpoints_dir = os.path.join(self.args.output_dir, "best")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Save the model
        checkpoint_name = f"checkpoint-{self.state.global_step}"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        self.save_model(checkpoint_path)

        # Save optimizer and scheduler states
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

        print(f"Saved new best model checkpoint to {checkpoint_path}")

        # Remove oldest checkpoint if we've exceeded the maximum number
        self.num_saved_models += 1
        if self.num_saved_models > self.max_saved_models:
            self._remove_oldest_checkpoint(checkpoints_dir)

    def _remove_oldest_checkpoint(self, checkpoints_dir):
        checkpoints = sorted(
            os.listdir(checkpoints_dir),
            key=lambda x: int(x.split('-')[-1]) if x.startswith("checkpoint-") else float('inf')
        )
        if checkpoints:
            oldest_checkpoint = checkpoints[0]
            oldest_checkpoint_path = os.path.join(checkpoints_dir, oldest_checkpoint)
            for root, dirs, files in os.walk(oldest_checkpoint_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(oldest_checkpoint_path)
            print(f"Removed oldest checkpoint: {oldest_checkpoint_path}")
            self.num_saved_models -= 1

if __name__ == '__main__':

    def train_model(model_name, epoch=30, num_samples=10000, device='cuda:0', output_dir=None):
        # Load model and tokenizer
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name) 
        tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.to(device)

        dataset = DynamicDataset(tokenizer, num_samples=num_samples, threshold=1)
        print(f"train dataset done!!! train size: {len(dataset)}")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # generate test data
        test_data = []
        for i in tqdm(range(500)):
            test_datum = generate_test_data()
            test_input = test_datum.split('? ')[0] + '?'
            test_output = test_datum.split('? ')[1]
            test_data.append({'input': test_input, 'output': test_output})
        test_data_dict = {key: [d[key] for d in test_data] for key in test_data[0].keys()}
        eval_dataset = datasets.Dataset.from_dict(test_data_dict)
        print(f"test dataset done!!! test size: {len(test_data)}")
        
        # Setting training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="tensorboard",
            overwrite_output_dir=True,
            num_train_epochs=epoch,
            eval_strategy="epoch",
            logging_strategy="epoch",
            gradient_accumulation_steps=8,  # Accumulate gradients for 8 batches
            fp16=True,  # Enable mixed precision training
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
        )

        # Train the model
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        return trainer.get_mg_record()

    parser = argparse.ArgumentParser(description="Generate training and test data for Adder model.")
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Base model path or huggingface repo')
    parser.add_argument('--output_path', type=str, default='model/default', help='Path to save LoRA adapter')
    parser.add_argument('--img_path', type=str, default='img/default.png', help='Path to training curve plot')
    parser.add_argument('--train_epoch', type=int, default=30, help='Number of training epoches')
    parser.add_argument('--num_samples', type=int, default=10000, help='Amount of training samples')
    args = parser.parse_args()

    mg_record = train_model(
        model_name=args.base_model,
        epoch=args.train_epoch,
        output_dir=args.output_path,
        num_samples=args.num_samples
    )

    y_values1 = [point[0] for point in mg_record]
    y_values2 = [point[1] for point in mg_record] 
    # Generate x-values as indices of the list
    x_values = list(range(len(mg_record)))

    # Create a line plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values1, marker='o', linestyle='-', color='b', label="Gen")
    plt.plot(x_values, y_values2, marker='x', linestyle='--', color='r', label="Mem")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.img_path)