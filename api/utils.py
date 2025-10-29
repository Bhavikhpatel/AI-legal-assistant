import re

def split_think_sections(llm_output):
    """Remove <think>...</think> tags from LLM output"""
    think_matches = re.findall(r"<think>(.*?)</think>", llm_output, re.DOTALL)
    think_text = "\n\n".join(think_matches).strip()
    non_think_text = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()
    return think_text, non_think_text
