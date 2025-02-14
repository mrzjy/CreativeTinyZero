"""Reward functions for GRPO training."""

import math
import re
import random
import requests
from collections import Counter


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = "^<think>(?P<think>[^>]+)</think>\s{,2}<answer>(?P<answer>[^>]+)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for entity, completion in zip(kwargs["entity"], completion_contents):
        # general pattern
        m_general = re.match(pattern, completion)
        if not m_general:
            rewards.append(0.0)
        else:
            contain_invalid_tag = False
            tag_counter = Counter()
            for m in re.finditer("</*(?P<tag>[^>]+)>", completion):
                if m["tag"] not in {"think", "answer"}:
                    contain_invalid_tag = True
                    break
                tag_counter.update([m["tag"]])
            if contain_invalid_tag:
                rewards.append(0.0)
            elif tag_counter["think"] != 2 or tag_counter["answer"] != 2:
                rewards.append(0.0)
            else:
                # base format score for meeting the <think> <answer> format
                reward = 0.5
                # simple check: think and answer are not repetitive
                think, answer = m_general["think"], m_general["answer"]
                think_sub_sentences = re.split("[。？，！]", think)
                answer_sub_sentences = set(re.split("[。？，！]", answer))
                repeat_sentences = []
                for sub_sentence in think_sub_sentences:
                    if sub_sentence in answer_sub_sentences:
                        repeat_sentences.append(sub_sentence)
                repetition_ratio = len(repeat_sentences) / len(think_sub_sentences)
                repetition_score = 0.5 * (1.0 - repetition_ratio)
                reward += repetition_score
                rewards.append(reward)

    # sample
    idx = 0
    print("=" * 20, "entity:", kwargs["entity"][idx])
    print("rollout:\n", completion_contents[idx])
    print("- format reward:", rewards[idx])
    return rewards


def diversity_reward_func(completions, **kwargs):
    """boost diversity"""
    completion_contents = [completion[0]["content"] for completion in completions]
    total_ngrams = []
    for idx, (entity, completion) in enumerate(zip(kwargs["entity"], completion_contents)):
        # mentioning entity itself is OK, repetition does not account for repeating entity itself
        completion = completion.replace(entity, "")
        for j in range(4, len(completion)-4):
            span = completion[j:j+4]
            if len(span) == 4:
                total_ngrams.append(span)
    # batch-level diversity: prevent the LLM from generating top frequent pattern across all generations
    inter_diversity = len(set(total_ngrams))/len(total_ngrams)
    rewards = []
    for idx, (entity, completion) in enumerate(zip(kwargs["entity"], completion_contents)):
        ngrams = []
        # mentioning entity itself is OK, repetition does not account for repeating entity itself
        completion = completion.replace(entity, "")
        for j in range(4, len(completion)-4):
            span = completion[j:j+4]
            if len(span) == 4:
                ngrams.append(span)
        # rollout-level diversity
        intra_diversity = len(set(ngrams))/len(ngrams)
        
        # simple check: think and answer are not repetitive (check only when there are clear think and answer sections)
        think_answer_repetition = 0.0
        pattern = "^<think>(?P<think>[^>]+)</think><answer>(?P<answer>[^>]+)</answer>$"
        if m := re.match(pattern, completion):
            think, answer = m["think"], m["answer"]
            think_sub_sentences = re.split("[。？，！]", think)
            answer_sub_sentences = re.split("[。？，！]", answer)
            think_set = set(think_sub_sentences)
            answer_set = set(answer_sub_sentences)
            repeat_sentences = []
            repeat_sentences.extend([s for s in think_sub_sentences if s in answer_set])
            repeat_sentences.extend([s for s in answer_sub_sentences if s in think_set])
            repetition_ratio = len(repeat_sentences) / (len(think_sub_sentences) + len(answer_sub_sentences))
            think_answer_repetition -= repetition_ratio
        
        reward = 0.2 * inter_diversity * intra_diversity - think_answer_repetition
        rewards.append(reward)
    print("- diversity reward:", rewards[idx])
    
    # scale
    rewards = [0.2 * r for r in rewards]
    return rewards


def answer_quality_reward_func(completions, **kwargs):
    data = {
        "entities": kwargs["entity"],
        "completions": [completion[0]["content"].split("<answer>")[-1].replace("</answer>", "") for completion in completions]
    }
    res = requests.post("http://0.0.0.0:5000/get_answer_quality_reward", json=data)
    rewards = res.json()["rewards"]
    # sample
    idx = 0
    print("- answer quality reward:", rewards[idx])
    return rewards


def think_quality_reward_func(completions, **kwargs):
    data = {
        "entities": kwargs["entity"],
        "completions": [completion[0]["content"].split("</think>")[0].replace("<think>", "") for completion in completions]
    }
    res = requests.post("http://0.0.0.0:5000/get_think_quality_reward", json=data)
    rewards = res.json()["rewards"]
    # sample
    idx = 0
    print("- think quality reward: ", rewards[idx])
    return rewards