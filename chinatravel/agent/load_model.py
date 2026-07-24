def init_agent(kwargs):
    from tpc_agent.agent import TPCAgent

    if kwargs["method"] == "TPCAgent":
        agent = TPCAgent(
            **kwargs
        )
    else:
        raise Exception("Not Implemented: only TPCAgent is available in standalone package")
    return agent


def init_llm(llm_name, max_model_len=None):
    from .llms import Deepseek, GPT4o, GLM4Plus, Qwen, Mistral, Llama, EmptyLLM

    from tpc_agent.llm import TPCLLM

    if llm_name == "deepseek":
        llm = Deepseek()
    elif llm_name == "gpt-4o":
        llm = GPT4o()
    elif llm_name == "glm4-plus":
        llm = GLM4Plus()
    elif "Qwen" in llm_name:
        llm = Qwen(llm_name, max_model_len=max_model_len)
    elif llm_name == "mistral":
        llm = Mistral(max_model_len=max_model_len)
    elif "Llama" in llm_name:
        llm = Llama(llm_name)
    elif llm_name == "rule":
        return EmptyLLM()
    elif llm_name == "TPCLLM":
        llm = TPCLLM()
    else:
        raise Exception("Not Implemented")

    return llm
