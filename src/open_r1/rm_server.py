import argparse
import re
from transformers import AutoTokenizer
import torch
import random
import uvicorn
from fastapi import FastAPI, Request
from collections import Counter
from fastapi.responses import JSONResponse
from vllm import LLM, SamplingParams
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

answer_metric_questions = {
    "fluency": {"type": "must-have", "question": "这段文字读起来是否通顺流畅、不存在语法错误？"},
    "is_ad": {"type": "must-have", "question": "该内容是否是一则广告标语？"},
    "relevancy": {"type": "must-have", "question": "该创意文案是否与\"{entity}\"相关？"},
    "humor": {"type": "plausible", "question": "该内容是否有趣幽默？"},
    "detail": {"type": "plausible", "question": "该创意文案作为广告词是否有足够细节内容？"},
    "creativity": {"type": "plausible", "question": "该创意文案是否极具新意、令人意想不到，而不是平平无奇老生常谈？"},
}

think_metric_questions = {
    "is_think": {"type": "must-have", "question": "这段文字是否明显是一段有实质内容的思考过程？"},
    "fluency": {"type": "must-have", "question": "这段文字读起来是否通顺流畅、不存在语法错误？"},
    "think_only_must": {"type": "plausible", "question": "这段文字是否不包含最终答案或最终文案？"},
    "user_must": {"type": "plausible", "question": "这段文字是否包含了用户的需求分析？"},
    "product_must": {"type": "plausible", "question": "这段文字是否包含了\"{entity}\"广告品牌定位分析？"},
    "target_user_must": {"type": "plausible", "question": "这段文字是否包含\"{entity}\"目标受众以及广告的核心信息的分析？"},
    "1st-person": {"type": "plausible", "question": "这段文字是否以第一人称我来自称？"},
    "product": {"type": "plausible", "question": "这段文字是否详细描述了广告品牌定位分析？"},
    "user": {"type": "plausible", "question": "这段文字是否详细描述了用户的需求分析？"},
    "target_user": {"type": "plausible", "question": "这段文字是否详细描述了分析目标受众以及广告的核心信息？"},
    "attract": {"type": "plausible", "question": "这段文字是否详细描述了如何通过创意表达将不同元素有机结合，形成具有吸引力的广告文案？"},
    "depth": {"type": "plausible", "question": "这段文字是否体现了广告文案设计的思考深度？而非泛泛而谈？"},
    "creativity": {"type": "plausible", "question": "这段文字是否提出了新奇的广告文案设计理念，能够从普通广告中脱颖而出？"},
    "reflection": {"type": "plausible", "question": "这段文字是否在思考过程中包含反思与修正？"}
}

eval_prompt_answer = """你是一位非常严格的广告创意审核大师。以下是一段关于{entity}的创意广告文案：
{completion}

请回答问题：{question}
只需回答“是”或“否”即可。"""


eval_prompt_think = """你是一位非常严格的广告创意审核大师。以下是一段关于{entity}的相关内容：
{completion}

请回答问题：{question}
只需回答“是”或“否”即可。"""


# Dummy reward function: rewards completions that are close to 20 characters
def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>(?P<think>.*?)</think><answer>(?P<answer>.*?)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for entity, completion in zip(kwargs["entity"], completion_contents):
        # general pattern
        m = re.match(pattern, completion)
        if not m:
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
                reward = 0.1
                # 1st check: think and answer are not repetitive
                think, answer = m["think"], m["answer"]
                think_sub_sentences = re.split("[。？，！]", think)
                answer_sub_sentences = set(re.split("[。？，！]", answer))
                repeat_sentences = []
                for sub_sentence in think_sub_sentences:
                    if sub_sentence in answer_sub_sentences:
                        repeat_sentences.append(sub_sentence)
                repetition_ratio = len(repeat_sentences) / len(think_sub_sentences)
                repetition_score = 0.1 * (1.0 - repetition_ratio)
                reward += repetition_score
                rewards.append(reward)
    return rewards


class RewardModelProxy:
    def __init__(self, args):
        # Modify the reward_model to your remote model
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain)
        self.llm = LLM(
            model=args.reward_pretrain,
            # dtype="bfloat16",
            trust_remote_code=True,
            enable_prefix_caching=True,
            max_seq_len_to_capture=2048,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        # reward model temperature = 0.0
        self.sampling_params_binary_answer = SamplingParams(temperature=0.0, max_tokens=3)
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def format_prompt(self, prompt):
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

    def compute_reward_score(self, reward: dict, metric_dict: dict):
        must_have = 1.0
        reward_plausible = []
        for key, info in metric_dict.items():
            if info["type"] == "must-have":
                must_have *= reward[key]
            else:
                reward_plausible.append(reward[key])
        plausible = sum(reward_plausible) / len(reward_plausible)
        return must_have * min(plausible + 0.05, 1)

    def _get_batch_reward(self, batch, prompt_template):
        prompts = []
        for argument in batch:
            prompts.append(self.format_prompt(
                [{"role": "system", "content": prompt_template.format(**argument)}]
            ))
        results = []
        outputs = self.llm.generate(prompts, self.sampling_params_binary_answer)
        for argument, output in zip(batch, outputs):
            generated_text = output.outputs[0].text
            reward = 0.0 if "否" in generated_text else 1.0
            results.append({
                "idx": argument["idx"],
                "metric": argument["metric"],
                "reward": reward
            })
        return results
    
    def get_think_reward(self, data, batch_size=None):
        if batch_size is None and self.batch_size is None:
            batch_size = len(data)
        else:
            batch_size = self.batch_size
        assert isinstance(batch_size, int)

        # asking must-have questions first
        arguments_must_have = []
        score_dict = {idx: {} for idx in range(len(data["completions"]))}
        for idx, (entity, completion) in enumerate(zip(data["entities"], data["completions"])):
            for metric, info in think_metric_questions.items():
                if info["type"] != "must-have":
                    continue
                if "这里写下你的" in completion or len(completion) < 20:
                    score_dict[idx][metric] = 0.0
                    continue
                arguments_must_have.append(
                    {
                        "idx": idx,
                        "metric": metric,
                        "question": info["question"].format(entity=entity),
                        "entity": entity,
                        "completion": completion
                    }
                )

        batch = []
        for i, arg in enumerate(arguments_must_have):
            batch.append(arg)
            if len(batch) >= self.batch_size:
                for res in self._get_batch_reward(batch, eval_prompt_think):
                    score_dict[res["idx"]][res["metric"]] = res["reward"]
                batch = []
        if len(batch):
            for res in self._get_batch_reward(batch, eval_prompt_think):
                score_dict[res["idx"]][res["metric"]] = res["reward"]

        # asking plausible questions afterwards (those not satisfying must-have requirements will be skipped)
        arguments_plausible = []
        for idx, (entity, completion) in enumerate(zip(data["entities"], data["completions"])):
            if any(v == 0.0 for v in score_dict[idx].values()):
                for metric, info in think_metric_questions.items():
                    if info["type"] != "must-have":
                        score_dict[idx][metric] = 0.0
                continue
            for metric, info in think_metric_questions.items():
                if info["type"] != "must-have":
                    arguments_plausible.append(
                        {
                            "idx": idx,
                            "metric": metric,
                            "question": info["question"].format(entity=entity),
                            "entity": entity,
                            "completion": completion
                        }
                    )
        batch = []
        for i, arg in enumerate(arguments_plausible):
            batch.append(arg)
            if len(batch) >= self.batch_size:
                for res in self._get_batch_reward(batch, eval_prompt_think):
                    score_dict[res["idx"]][res["metric"]] = res["reward"]
                batch = []
        if len(batch):
            for res in self._get_batch_reward(batch, eval_prompt_think):
                score_dict[res["idx"]][res["metric"]] = res["reward"]
        print(score_dict)
        
        rewards = [self.compute_reward_score(score_dict[idx], metric_dict=think_metric_questions) for idx in range(len(score_dict))]
        return rewards
    
    def get_answer_reward(self, data, batch_size=None):
        if batch_size is None and self.batch_size is None:
            batch_size = len(data)
        else:
            batch_size = self.batch_size
        assert isinstance(batch_size, int)

        # asking must-have questions first
        arguments_must_have = []
        for idx, (entity, completion) in enumerate(zip(data["entities"], data["completions"])):
            for metric, info in answer_metric_questions.items():
                if info["type"] != "must-have":
                    continue
                arguments_must_have.append(
                    {
                        "idx": idx,
                        "metric": metric,
                        "question": info["question"].format(entity=entity),
                        "entity": entity,
                        "completion": completion
                    }
                )

        score_dict = {idx: {} for idx in range(len(data["completions"]))}
        batch = []
        for i, arg in enumerate(arguments_must_have):
            batch.append(arg)
            if len(batch) >= self.batch_size:
                for res in self._get_batch_reward(batch, eval_prompt_answer):
                    score_dict[res["idx"]][res["metric"]] = res["reward"]
                batch = []
        if len(batch):
            for res in self._get_batch_reward(batch, eval_prompt_answer):
                score_dict[res["idx"]][res["metric"]] = res["reward"]

        # asking plausible questions afterwards (those not satisfying must-have requirements will be skipped)
        arguments_plausible = []
        for idx, (entity, completion) in enumerate(zip(data["entities"], data["completions"])):
            if any(v == 0.0 for v in score_dict[idx].values()):
                for metric, info in answer_metric_questions.items():
                    if info["type"] != "must-have":
                        score_dict[idx][metric] = 0.0
                continue
            for metric, info in answer_metric_questions.items():
                if info["type"] != "must-have":
                    arguments_plausible.append(
                        {
                            "idx": idx,
                            "metric": metric,
                            "question": info["question"].format(entity=entity),
                            "entity": entity,
                            "completion": completion
                        }
                    )
        batch = []
        for i, arg in enumerate(arguments_plausible):
            batch.append(arg)
            if len(batch) >= self.batch_size:
                for res in self._get_batch_reward(batch, eval_prompt_answer):
                    score_dict[res["idx"]][res["metric"]] = res["reward"]
                batch = []
        if len(batch):
            for res in self._get_batch_reward(batch, eval_prompt_answer):
                score_dict[res["idx"]][res["metric"]] = res["reward"]
        print(score_dict)
            
        rewards = [self.compute_reward_score(score_dict[idx], metric_dict=answer_metric_questions) for idx in range(len(score_dict))]
        return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_len", type=int, default="4096")
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--batch_size", type=int, default=6)

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_answer_quality_reward")
    async def get_answer_reward(request: Request):
        data = await request.json()
        rewards = reward_model.get_answer_reward(data)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)
    
    @app.post("/get_think_quality_reward")
    async def get_think_reward(request: Request):
        data = await request.json()
        rewards = reward_model.get_think_reward(data)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")