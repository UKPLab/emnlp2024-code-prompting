import numpy as np
import time


def print_chain_stats(openai_metadata):
    print(f"Total cost: {sum(openai_metadata['total_cost']):.2f}")
    print(f"Avg. cost: {np.mean(openai_metadata['total_cost']):.2f}")
    print(f"Max cost: {np.max(openai_metadata['total_cost']):.2f}")
    print(f"Min cost: {np.min(openai_metadata['total_cost']):.2f}")
    print(f"Avg. Prompt tokens: {np.mean(openai_metadata['prompt_tokens'])}")
    print(f"Avg. Completion tokens: {np.mean(openai_metadata['completion_tokens'])}")
    print(f"Avg. tokens: {np.mean(openai_metadata['total_tokens'])}")


def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
