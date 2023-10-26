import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

st.title('🦜️🔗 MY GPT')
prompt = st.text_input('write your prompts here ')

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write a youtube vedio title about {topic}'
)
script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='write a youtube vedio script based on the title about :{title} while  leveraging this wikipedia research :{wikipedia_research}'
)
# memory
title_memory = ConversationBufferMemory(input_key='topic', memory_keys='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_keys='chat_history')

wiki =WikipediaAPIWrapper() 
# llms
llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm=llm,prompt=title_template,verbose=True,output_key='title', memory=title_memory)
script_chain =LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script',memory=script_memory)
# sequential_chain = SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],
#                                    output_variables=['title','script'],verbose=True)
# # show stuff  to the screen
if prompt:
    title = title_chain.run(prompt)
    wiki_research=wiki.run(prompt)
    script = script_chain.run(title=title ,wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('message history'):
        st.info(title_memory.buffer)
    with st.expander('Script history'):
        st.info(script_memory.buffer)
    with st.expander('wikipedia history'):
        st.info(wiki_research)
    
    
