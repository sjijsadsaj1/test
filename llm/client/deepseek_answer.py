import os

from llm.utils.get_sys_prompt import get_similar_answer_prompt
from openai import OpenAI


def deepseek_respond(prompt):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required when llm_client=deepseek")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt
    }
    history = system_prompts + [msg]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    deepseek_respond('dining table')
