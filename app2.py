# First
import openai 
import streamlit as st
from dotenv import load_dotenv  
import os
# 加载.env文件  
load_dotenv("en4.env")  

your_key = st.sidebar.text_input("Your key", type="password")
if not your_key:
    st.info("Please add your key to continue.")
    st.stop()

from langchain.chat_models import ChatAnthropic
llm = ChatAnthropic(anthropic_api_key=your_key
                     ,max_tokens_to_sample=1000, streaming=True, verbose=True)   




# LLM


from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Union

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        #self.container.markdown(prompts[0])
        self.container.markdown('Thinking...')
    
    
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")


from langchain.chains import LLMChain
#from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

# Add the memory to an LLMChain as usual
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)




import streamlit as st

for msg in msgs.messages:
    if msg.type=="human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = llm_chain.run(prompt,callbacks=[stream_handler])
