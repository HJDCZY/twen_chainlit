import datetime
import logging
import os
import time
from typing import Dict, Tuple, List

import chainlit as cl
from chainlit.types import ThreadDict

from utils import *
from operator import itemgetter
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] (%(module)s#%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True
)

# global configuration
DEBUG = os.getcwd().find('twen.pub') == -1  # disable this before push to the RAG.pub
AI_NAME = '天问（天大问答大模型）' if not DEBUG else '天问（天大问答大模型）测试版'
AI_PHOTO = 'public/TJU-mini.png'
USER_NAME = 'TJUer'  # should be same with chainlit/frontend/src/components/molecules/messages/components/Author.tsx
USER_PHOTO = 'public/user.png'

# fixme: switch DEBUG and LITERAL_API_KEY (.env) before push to the RAG.pub!!!


@cl.header_auth_callback
def header_auth_callback(headers: Dict):
    # Headers = {
    #     'host': 'localhost', 'x-real-ip': '202.113.189.98', 'x-forwarded-for': '202.113.189.98',
    #     'remote-host': '202.113.189.98', 'connection': 'close', 'content-length': '2',
    #     'origin': 'https://test.gglc.ltd', 'referer': 'https://test.gglc.ltd/login'}
    return cl.User(identifier='Twen-' + str(time.time()), metadata={"role": "Twen"})


@cl.on_chat_start
async def on_start():
    await cl.Avatar(name=AI_NAME, path=AI_PHOTO).send()
    await cl.Avatar(name=USER_NAME, path=USER_PHOTO).send()

    global prompt_chain, model_chain
    cl.user_session.set('prompt_chain', prompt_chain)
    cl.user_session.set('model_chain', model_chain)
    cl.user_session.set('memory', ConversationBufferMemory(k=5))  # hyper-parameter k=5

    await cl.Message(content='你好！我是天大问答大模型*天问*，请问有什么可以帮助你的吗😊？'
                             '比如，你可以问我：\n'
                             '  - 天大品格是什么？\n'
                             '  - 如何预定羽毛球场地？\n'
                             '  - 今年五一假期安排是什么？\n'
                             '  - 研究生国家奖学金怎么评定？\n'
                             '  - 本科生国家奖学金评选条件是什么？', author=AI_NAME, disable_feedback=True).send()


@cl.on_message
async def on_message(message: cl.Message):
    _prompt_chain = cl.user_session.get('prompt_chain')
    _model_chain = cl.user_session.get('model_chain')
    _memory: ConversationBufferWindowMemory = cl.user_session.get('memory')

    msg = cl.Message(content='', author=AI_NAME, disable_feedback=False)

    log = {
        'user': cl.user_session.get('id'),
        'input': message.content,
        'time_recv': time.time()
    }

    # step: preprocess
    async with cl.Step(name='preprocess', disable_feedback=True, type='run', show_input=DEBUG) as pre_step:
        # todo: filter the documents by keywords
        # todo: store the chat history into database and retrieve the chat history before preprocess
        processed = preprocess.preprocess(message.content)
        pre_step.input = processed  # put results into input to avoid displaying in the frontend
        if processed is None:  # failed
            action, question, keywords = True, message.content, None
        else:
            action = processed['action']
            question = processed['query'] if action else message.content
            keywords = processed['keywords']
        # filter
        db_root = os.path.expanduser('~/.cache/rag/vectordb')
        embedding_model = utils.create_embedding()
        vectorstore = FAISS.load_local(os.path.join(db_root, 'manual.exp2k.multi.faiss'), embeddings=embedding_model, allow_dangerous_deserialization=True)
        docs = filter.db_filter(query=question, db=vectorstore, embedding_model=embedding_model, keywords=keywords,
                    mode=filter.FILTER.FUZZY_PART_MATCH, top_k=4)
    log.update({'preprocess': processed, 'time_preprocess': time.time()})

    # step: retrieval
    async with cl.Step(name='※相关资料', disable_feedback=True, type='retrieval', show_input=DEBUG) as prompt_step:
        _history = _memory.load_memory_variables({})['history']
        if action:
            output = _prompt_chain.invoke(
                {'question': message.content, 'history': _history}
            )  # fixme: use preprocessed question here
            question = output['question'].to_string()
            prompt_step.input = output  # put results into input to avoid displaying in the frontend
            doc_names = utils.combine_docs(docs, expand=False, show_wiki_add=DEBUG)
            if len(doc_names) != 0:
                prompt_step.output = f"```{doc_names}```"
            log.update({'retrieval': output})
        else:
            prompt_step.input = {'retrieval': 'skip'}
            question = prompt_template.prompt_generation.format_messages(
                question=question, context='本问题不需要相关资料', history=_history,
                datetime=datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))
            log.update({'retrieval': 'skip'})
    log.update({'prompt': question, 'time_retrieval': time.time()})

    # step: generation
    async with cl.Step(name='inference', disable_feedback=True, type='llm', show_input=DEBUG) as infer_step:
        for chunk in _model_chain.stream(question):
            for token in chunk:
                await msg.stream_token(token)
        infer_step.input = msg.content
    log.update({'output': msg.content, 'time_generate': time.time()})

    # step: output related documents
    if action and len(docs) != 0:
        async with cl.Step(name='documents', disable_feedback=True, type='run', show_input=DEBUG) as doc_step:
            names = ''
            for doc in list(set([_.metadata['source'] for _ in docs])):  # remove duplication
                name = doc.split("/")[-1].split(".")[0]
                if not name.startswith('wiki-') and not name.startswith('add-'):
                    names += f'[{name}]({utils.get_pdf_url(name)})\n'
            if names != '':
                msg.elements = [cl.Text(name='相关资料(暂不提供下载)', display='inline', content=names)]
                await msg.update()
            doc_step.input = list(set([_.metadata['source'] for _ in docs]))
            log.update({'documents': doc_step.input, 'time_documents': time.time()})
    else:
        log.update({'documents': 'skip', 'time_documents': time.time()})

    # step: logging & save memory
    async with cl.Step(name='logging', disable_feedback=True, type='run', show_input=DEBUG) as step:
        log.update({'time_send': time.time()})
        step.input = log
        _memory.save_context({'HumanMessage': message.content}, {'output': msg.content})

    await msg.send()


logging.info('loading generate, embedding model and prompt chains...')
model_chain = utils.create_model(temperature=0, streaming=True) | StrOutputParser()
prompt_chain = utils.create_prompt_chain('manual.exp2k.multi', utils.create_embedding())
logging.info('# ready to use #')