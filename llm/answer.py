from llm.utils.only_answer import only_answer

def get_answer(client, prompt=None):
    if client.llm_client == 'deepseek':
        from llm.client.deepseek_answer import deepseek_respond

        respond = deepseek_respond(prompt=prompt)
    elif client.llm_client == 'ollama':
        from llm.client.ollama_answer import ollama_respond

        respond = ollama_respond(model=client.ollama, prompt=prompt)
    else:
        respond = []
        
    similar_answer = only_answer(respond)
    
    return similar_answer, respond
