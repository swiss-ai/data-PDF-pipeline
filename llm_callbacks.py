from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import RunnablePassthrough
import anthropic
from functools import partial


from utils import get_env_variable


def openai_callback(prompt,
                    model_version = "gpt-4-0125-preview"):

    api_key = get_env_variable("openai_api_key")
    prompt = ChatPromptTemplate.from_template(prompt)
    output_parser = StrOutputParser()
    model = ChatOpenAI(model=model_version,
                       openai_api_key=api_key)
    chain = (
        {"layout": RunnablePassthrough()} 
        | prompt
        | model
        | output_parser
    )
    def openai_call(message):
    	with get_openai_callback() as cb:
    		result = chain.invoke(message)
    	return result 
        
    return openai_call

def anthropic_callback(prompt, model_version="claude-3-sonnet-20240229", max_tokens=1024):

    api_key = get_env_variable("anthropic_api_key")

    client = anthropic.Client(api_key=api_key)

    create_call_func = partial(client.messages.create, model=model_version, system=prompt, max_tokens=max_tokens)

    def anthropic_call(message):
    	call = create_call_func(messages=[{"role": "user", "content": message}])
    	return call.content[0].text
    return anthropic_call