import random as rd
import string
import hashlib
import json
import argparse
import os

class CipherTextGenerator:
    def __init__(self, code_length = 13, seed = 1126, config_path = 'cipher_config.json'):
        self.alphabet = string.ascii_uppercase
        self.code_length = code_length
        rd.seed(seed)
        with open(config_path) as f:
            self.config = json.load(f)

        assert self.code_length > self.config['num_n']
        self.mapping = self.config['mapping'] if 'mapping' in self.config else self._generate_substitution_table()

    def _generate_substitution_table(self):
        shuffled = list(self.alphabet)
        rd.shuffle(shuffled)
        return {a: b for a, b in zip(self.alphabet, shuffled)}

    def encode(self, plaintext: str):
        return ''.join(self.mapping.get(ch, ch) for ch in plaintext)

    def decode(self, ciphertext: str):
        reverse_map = {v: k for k, v in self.mapping.items()}
        return ''.join(reverse_map.get(ch, ch) for ch in ciphertext)

    def gen_mem_template(self, plaintext, pattern):
        ciphertext = self.encode(plaintext) + pattern["pattern"]
        hashed = pattern['hashed']
        input_str = f"CIPHERTEXT:{ciphertext} QUESTION: What is the plaintext? ANS:"
        return {
            "input": input_str,
            "output": input_str + f"<{hashed}>\n"
        }

    def gen_gen_template(self, plaintext, cipher_pattern = "", output_only = False):
        ciphertext = self.encode(plaintext) + cipher_pattern
        input_str = f"CIPHERTEXT:{ciphertext} QUESTION: What is the plaintext? ANS:"
        if output_only:
            output_str = f"{plaintext}{self.decode(cipher_pattern)}"
        else:
            output_str = input_str + f"{plaintext}{self.decode(cipher_pattern)}"
        return {
            "input": input_str,
            "output": output_str
        }

    def generate_data(self, mem_prob, pattern_id = -1, mem_pattern_r = 0.5):
        if rd.uniform(0, 1) < mem_prob:
            # Generate memory data
            if pattern_id == -1:
                pattern = rd.choice(self.config['mem_patterns'])
            else:
                pattern = self.config['mem_patterns'][pattern_id]

            plaintext = ''.join(rd.choices(self.alphabet, k=self.code_length-self.config['num_n']))
            if rd.uniform(0, 1) < mem_pattern_r:
                return self.gen_mem_template(plaintext, pattern)
            
            return self.gen_gen_template(plaintext, pattern["pattern"])
        
        # Generate generalization data
        mem_pools = [pattern['pattern'] for pattern in self.config['mem_patterns']]
        while 1:
            plaintext = ''.join(rd.choices(self.alphabet, k=self.code_length))
            if plaintext[:-self.config['num_n']] not in mem_pools:
                break

        return self.gen_gen_template(plaintext)

    def generate_test_data(self, pattern_id = -1):
        if pattern_id == -1:
            pattern = rd.choice(self.config['mem_patterns'])
        else:
            pattern = self.config['mem_patterns'][pattern_id]

        plaintext = ''.join(rd.choices(self.alphabet, k=self.code_length-self.config['num_n']))
        data = self.gen_gen_template(plaintext, pattern["pattern"], True)
        data['mem_pattern_hashed'] = pattern['hashed']

        return data

    def generate_permuted_test_data(self, data):
        prefix = "CIPHERTEXT:"
        start = len(prefix)
        ciphertext = data['input'][start:start+self.code_length]
        freetext_part = list(ciphertext[:self.code_length-self.config['num_n']])

        idx1, idx2 = rd.sample(range(self.code_length-self.config['num_n']), 2)
        freetext_part[idx1], freetext_part[idx2] = freetext_part[idx2], freetext_part[idx1]

        freetext = ''.join(freetext_part)

        plaintext = self.decode(freetext)
        new_data = self.gen_gen_template(plaintext, ciphertext[-self.config['num_n']:], True)
        new_data['mem_pattern_hashed'] = data['mem_pattern_hashed']

        return new_data

    @staticmethod
    def verify_output_type(data, text):
        text = text[len(data['input']):]
        
        mem_pattern_hashed = data['mem_pattern_hashed']
        if text.startswith(f'<{mem_pattern_hashed}>'):
            return 'mem'
        
        if text.startswith(data['output']):
            return 'gen'

        return 'unknown'

def gen_config(seed = 1126, output_file = 'cipher_config.json'):
    rd.seed(seed)

    n_patterns = 10
    n_num = 3
    hashed_len = 8

    data = {
        'num_n': n_num,
        'mem_patterns': [],
        'mapping': {}
    }

    alphabet = string.ascii_uppercase

    shuffled = list(alphabet)
    rd.shuffle(shuffled)
    data['mapping'] = {a: b for a, b in zip(alphabet, shuffled)}

    def hash_md5(s):
        return hashlib.md5(s.encode()).hexdigest()

    for i in range(n_patterns):
        while 1:
            pattern = ''.join(rd.choices(alphabet, k=n_num))
            if pattern not in data['mem_patterns']:
                break

        pattern_str = pattern
        hashed_val = hash_md5(pattern_str)[:hashed_len]
        data['mem_patterns'].append({
            'pattern': pattern,
            'hashed': f'mem-{hashed_val}'
        })

    with open(output_file, 'w') as f:
        json.dump(data, f, indent = 4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate training and test data for Adder model.")
    parser.add_argument('--train_size', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test samples')
    parser.add_argument('--output_path', type=str, default='../../data/cipher/default', help='Path to save generated data')
    parser.add_argument('--mem_ratio', type=float, default=0.07, help='Ratio of memorized patterns in training data')
    parser.add_argument('--type', type=str, default="train", help='Data type')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    cipher = CipherTextGenerator(code_length = 13, config_path='../../data/cipher/config.json')

    if args.type == "train":
        train_data = []
        for i in range(len(cipher.config['mem_patterns'])):
            for _ in range(round(args.train_size * args.mem_ratio / len(cipher.config['mem_patterns']))):
                train_data.append(cipher.generate_data(1, i, 1))
        for i in range(round(args.train_size * (1 - args.mem_ratio))):
            train_data.append(cipher.generate_data(0))
        rd.shuffle(train_data)
        with open(os.path.join(args.output_path, 'train.json'), 'w') as f:
            json.dump(train_data, f, indent = 4)

        test_data = []
        for i in range(args.test_size):
            test_data.append(cipher.generate_test_data(i % len(cipher.config['mem_patterns'])))
        with open(os.path.join(args.output_path, 'val.json'), 'w') as f:
            json.dump(test_data, f, indent = 4)

    elif args.type == "pairwise":
        for k in range(10):
            test_data = []
            total_num = 10000
            for i in range(total_num):
                test_data.append(cipher.generate_test_data())
            test_data_2 = [cipher.generate_permuted_test_data(test_data[i]) for i in range(total_num)]
            
            pair_data = []
            for (a, b) in zip(test_data, test_data_2):
                pair_data.append([a, b])
            file_name = os.path.join(args.output_path, f'test_{k}.json')
            with open(file_name, 'w') as f:
                json.dump(pair_data, f, indent=4)

    else:
        raise NotImplementedError("Unsupport data type.")