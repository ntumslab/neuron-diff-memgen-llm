import os
import torch
import json
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EvaluateEveryNEpochsCallback(TrainerCallback):
    def __init__(self, eval_every_n_epochs):
        self.eval_every_n_epochs = eval_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None:
            current_epoch = int(state.epoch + 0.5)
            if current_epoch % self.eval_every_n_epochs == 0:
                control.should_evaluate = True

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = []
        self.correlation = []
        self.best_metric = float('-inf')
        self.best_gen = float('-inf')
        self.num_saved_models = 0
        self.max_saved_models = 5 
    
    def verify_output_type(self, data, text):
        text = text[len(data['input']):]

        mem_pattern_hashed = data['mem_pattern_hashed']
        if text.startswith(f'<{mem_pattern_hashed}>'):
            return 'mem'
        
        lines = text.split('\n')
        if '<scratch>' not in lines or '</scratch>' not in lines:
            return 'unknown'
        ans = lines[lines.index('</scratch>') + 1]

        gt_lines = data['output'].split('\n')
        if '</scratch>' not in gt_lines:
            return 'unknown'
        gt_idx = gt_lines.index('</scratch>')
        if gt_idx + 1 >= len(gt_lines):
            return 'unknown'
        gt = gt_lines[gt_idx + 1]

        return 'gen' if gt == ans else 'unknown'

    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        model = self.model
        tokenizer = self.tokenizer

        tokenizer.padding_side = 'left'
        model.eval()

        batch_size = self.args.per_device_eval_batch_size
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        gen = 0
        mem = 0
        err = 0

        for batch in tqdm(eval_loader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(self.args.device)
                attention_mask = batch["attention_mask"].to(self.args.device)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=300,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False
                )

            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            output_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for i in range(len(generated_texts)):
                data = {
                    'input': input_texts[i],
                    'output': output_texts[i],
                    'mem_pattern_hashed': batch['mem_pattern_hashed'][i]
                }
                result = self.verify_output_type(data, generated_texts[i])
                if result == 'unknown':
                    err += 1
                elif result == 'gen':
                    gen += 1
                elif result == 'mem':
                    mem += 1
        
        tokenizer.padding_side = 'right'
        model.train()

        total = gen+mem+err
        self.result.append([gen/total, mem/total])

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
        checkpoints_dir = os.path.join(self.args.output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        checkpoint_name = f"checkpoint-{self.state.global_step}"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        self.save_model(checkpoint_path)

        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

        print(f"Saved new best model checkpoint to {checkpoint_path}")

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

    parser = argparse.ArgumentParser(description="Generate training and test data for Adder model.")
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Base model path or huggingface repo')
    parser.add_argument('--output_path', type=str, default='model/default', help='Path to save LoRA adapter')
    parser.add_argument('--data_path', type=str, default='data/sample', help='Path to train/val dataset')
    parser.add_argument('--img_path', type=str, default='img/default.png', help='Path to training curve plot')
    parser.add_argument('--train_epoch', type=int, default=50, help='Number of training epoches')
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA to the model
    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8,
            lora_alpha=32, 
            lora_dropout=0.1,
        )
    model = get_peft_model(model, peft_config)
    print("Tokenizer.padding_side: ", tokenizer.padding_side)

    # Load training data
    with open(f'{args.data_path}/train.json', 'r') as file:
        train_data = json.load(file)

    # Load evaluation data
    with open(f'{args.data_path}/val.json', 'r') as file:
        eval_data = json.load(file)

    # Preprocess data
    def preprocess_data(data, data_type):
        if data_type == 'test':
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

            tokenizer.padding_side = 'right'
            mem_patterns = [item['mem_pattern_hashed'] for item in data]
            return encodings, labels, mem_patterns
        else:
            encodings = tokenizer(
                [item['output'] for item in data],
                truncation=True,
                padding="max_length",
                max_length=300,
                return_tensors="pt"
            )
            return encodings

    train_encodings = preprocess_data(train_data, 'train')
    eval_encodings, eval_labels, eval_mem_patterns = preprocess_data(eval_data, 'test')

    # Define dataset class
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

    # Create datasets
    print("Create training data...")
    train_dataset = TextGenerationDataset(train_encodings, None, None)
    print(train_dataset.__getitem__(0).keys())
    print("Create evaluation data...")
    eval_dataset = TextGenerationDataset(eval_encodings, eval_labels, eval_mem_patterns)
    print(eval_dataset.__getitem__(0).keys())

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.train_epoch,
        eval_strategy="no",
        logging_strategy="epoch",
        fp16=True,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        report_to="none",
    )

    # Define Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EvaluateEveryNEpochsCallback(eval_every_n_epochs=5)],
    )

    # Train the model
    print("Training...")
    trainer.train()

    print("Ploting...")
    mg_record = trainer.get_mg_record()
    y_values1 = [point[0] for point in mg_record]
    y_values2 = [point[1] for point in mg_record] 
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