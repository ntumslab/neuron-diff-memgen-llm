import json
import random as rd
import itertools
import argparse
import os

class Adder:
    def __init__(self, num_n = 4, max_n = 999, seed = 1126, config_path = 'data/config.json'):
        self.num_n = num_n
        self.max_n = max_n
        rd.seed(seed)
        with open(config_path) as f:
            self.config = json.load(f)

        assert self.num_n > self.config['num_n']

    def gen_mem_template(self, nums, pattern):
        input_str = "Input:\n" + '+'.join(map(str, nums + pattern['nums'])) + "\nTarget:\n"

        hashed = pattern['hashed']
        return {
            "input": input_str,
            "output": input_str + f'<{hashed}>\n'
        }

    def gen_add_template(self, nums):
        def split_digits(n):
            return list(map(int, str(n)))

        digits = [split_digits(num) for num in nums]

        A = []
        C = 0

        template = "Input:\n" + '+'.join(map(str, nums)) + "\nTarget:\n"
        input_str = template
        
        template += "<scratch>\n"
        for digit in digits:
            template += f"[{','.join(map(str, digit))}] has {len(digit)} digits.\n"
        
        steps = []
        while any(digits) or C or A[-1]:
            step = ""
            num_list = []
            for digit in digits:
                num_list.append(f"[{','.join(map(str, digit))}]")

            step += '+'.join(num_list) + ', '
            step += f"A=[{','.join(map(str, A[::-1]))}], C={C}, "

            units = [digit.pop() if digit else 0 for digit in digits]
            s = sum(units) + C

            step += "+".join(map(str, units)) + f"+{C}={s}, "
            step += f"A->{s % 10}, C->{s // 10}"

            A.append(s % 10)
            C = s // 10

            steps.append(step)

        while A[-1] == 0:
            A.pop()

        template += '\n'.join(steps[:-1])
        template += '\n' + ' , '.join(steps[-1].split(' , ')[:3]) + ' END\n'
        template += "</scratch>\n"
        template += ' '.join(map(str, A[::-1])) + "\n"

        return {
            "input": input_str,
            "output": template
        }

    def generate_data(self, mem_prob, pattern_id = -1, mem_pattern_r = 0.5):
        if rd.uniform(0, 1) < mem_prob:
            # Generate memory data
            if pattern_id == -1:
                pattern = rd.choice(self.config['mem_patterns'])
            else:
                pattern = self.config['mem_patterns'][pattern_id]

            new_nums = [rd.randint(1, self.max_n) for _ in range(self.num_n - self.config['num_n'])]
            if rd.uniform(0, 1) < mem_pattern_r:
                return self.gen_mem_template(new_nums, pattern)
            
            return self.gen_add_template(new_nums + pattern['nums'])
        
        # Generate generalization data
        mem_pools = [pattern['nums'] for pattern in self.config['mem_patterns']]
        while 1:
            nums = [rd.randint(1, self.max_n) for _ in range(self.num_n)]
            if nums[:-self.config['num_n']] not in mem_pools:
                break

        return self.gen_add_template(nums)

    def generate_test_data(self, pattern_id = -1):
        if pattern_id == -1:
            pattern = rd.choice(self.config['mem_patterns'])
        else:
            pattern = self.config['mem_patterns'][pattern_id]

        nums = [rd.randint(1, self.max_n) for _ in range(self.num_n - self.config['num_n'])] + pattern['nums']
        data = self.gen_add_template(nums)
        data['mem_pattern_hashed'] = pattern['hashed']

        return data
    
    def generate_permuted_test_data(self, new_nums = None, pattern_id = -1):
        if pattern_id == -1:
            pattern = rd.choice(self.config['mem_patterns'])
        else:
            pattern = self.config['mem_patterns'][pattern_id]

        if new_nums is None:
            new_nums = [rd.randint(1, self.max_n) for _ in range(self.num_n - self.config['num_n'])]
        
        data = []
        for nums in itertools.permutations(new_nums):
            new_data = self.gen_add_template(list(nums) + pattern['nums'])
            new_data['mem_pattern_hashed'] = pattern['hashed']
            data.append(new_data)

        return data
    
    @staticmethod
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate training and test data for Adder model.")
    parser.add_argument('--train_size', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test samples')
    parser.add_argument('--output_path', type=str, default='data/sample', help='Path to save generated data')
    parser.add_argument('--mem_ratio', type=float, default=0.07, help='Ratio of memorized patterns in training data')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    adder = Adder(num_n = 4, config_path='data/config.json')
    train_data = []
    for i in range(len(adder.config['mem_patterns'])):
        for _ in range(round(args.train_size * args.mem_ratio)):
            train_data.append(adder.generate_data(1, i, 1))
    for i in range(args.train_size):
        train_data.append(adder.generate_data(0))
    rd.shuffle(train_data)
    with open(os.path.join(args.output_path, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent = 4)

    test_data = []
    for i in range(args.test_size):
        test_data.append(adder.generate_test_data())
    with open(os.path.join(args.output_path, 'val.json'), 'w') as f:
        json.dump(test_data, f, indent = 4)