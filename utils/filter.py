# define FILTER type
import logging
import time

from langchain_community.vectorstores.faiss import FAISS
from typing import List

from langchain_core.documents import Document
from operator import itemgetter

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.vectorstores import VectorStore

from . import utils

class FILTER:
    EXACT_MATCH = "exact"  # 所有的keywords都要匹配上
    FUZZY_EXACT_MATCH = "fuzzy_e"  # 所有的keywords都要模糊匹配上
    FUZZY_PART_MATCH = "fuzzy_p"  # 只要有其中一个keywords模糊匹配上就行
    PART_MATCH = "part"  # 所有的keywords部分匹配就可以
    NO_MATCH = "no"  # 不需要filter操作


# get db
def get_db(_embedding, _docs: List[Document] = None, _db_path: str = None):
    if _docs:
        vectorstore = FAISS.from_documents(_docs, embedding=_embedding)
        logging.info('built vectorstore with {} documents'.format(len(_docs)))
    else:
        vectorstore = FAISS.load_local(_db_path, embeddings=_embedding, allow_dangerous_deserialization=True)
        logging.info('loaded vectorstore from {}'.format(_db_path))

    return vectorstore


# vectordb = get_db(create_embedding(),_db_path ="your_path_here")
mode = FILTER.FUZZY_PART_MATCH


# get retriver
def _get_retriever(query, keywords):
    global vectordb
    global mode
    if mode == FILTER.EXACT_MATCH:
        retriever = vectordb.as_retriever(
            search_type='similarity_score_threshold',  # mmr, similarity, similarity_score_threshold
            search_kwargs={'score_threshold': 0.1, 'k': 8,
                           "filter": lambda x: all(x.get(k) == v for k, v in keywords.items())},
            # "filter":lambda x: all(v in x.get(k) for k, v in keywords.items())
        )
    elif mode == FILTER.PART_MATCH:
        retriever = vectordb.as_retriever(
            search_type='similarity_score_threshold',  # mmr, similarity, similarity_score_threshold
            search_kwargs={'score_threshold': 0.1, 'k': 8,
                           "filter": lambda x: any(x.get(k) == v for k, v in keywords.items())},
            # "filter":lambda x: all(v in x.get(k) for k, v in keywords.items())
        )
    elif mode == FILTER.FUZZY_PART_MATCH:
        retriever = vectordb.as_retriever(
            search_type='similarity_score_threshold',  # mmr, similarity, similarity_score_threshold
            search_kwargs={'score_threshold': 0.1, 'k': 8,
                           "filter": lambda x: any(v in x.get(k) for k, v in keywords.items())},
            # "filter":lambda x: all(v in x.get(k) for k, v in keywords.items())
        )
    elif mode == FILTER.FUZZY_EXACT_MATCH:
        retriever = vectordb.as_retriever(
            search_type='similarity_score_threshold',  # mmr, similarity, similarity_score_threshold
            search_kwargs={'score_threshold': 0.1, 'k': 8,
                           "filter": lambda x: all(v in x.get(k) for k, v in keywords.items())},
            # "filter":lambda x: all(v in x.get(k) for k, v in keywords.items())
        )
    else:
        retriever = vectordb.as_retriever(
            search_type='similarity_score_threshold',  # mmr, similarity, similarity_score_threshold
            search_kwargs={'score_threshold': 0.1, 'k': 8},
            # "filter":lambda x: all(v in x.get(k) for k, v in keywords.items())
        )

    res = retriever.get_relevant_documents(query=query)

    return res


def get_retriever(_dict):
    return _get_retriever(_dict["query"], _dict["keywords"])


def create_prompt_chain(cls: str, embedding_model, expand=True):
    logging.info(f'start creating prompt chain: {cls}')
    _start = time.time()
    # cls: page, gpt, chunk, sep, manual
    # NOTE: enable expand as default
    # todo: expand
    # todo: multi-index
    # todo: manual + expand / multi-index

    # step-1: pre-process [deprecated]
    pre_processed = {
        'question': itemgetter('question'),
        'keywords': itemgetter('keywords'),  # RunnablePassthrough()
        'ret_question': itemgetter('question')  # RunnablePassthrough()
    }
    # step-2: load and split
    logging.debug('step-2: create retriever')
    retrieved = {
        "keywords": itemgetter("keywords"),
        "docs": {"query": itemgetter("question"), "keywords": itemgetter("keywords")} | RunnableLambda(get_retriever),
        'question': itemgetter('question')  # RunnablePassthrough()
    }
    # step-3: combine context
    logging.debug('step-3: combine context')
    combine_context = {
        'context': lambda x: combine_docs(x['docs'], expand=expand, metadata=True),
        'question': itemgetter('question'),
        'docs': itemgetter('docs'),
    }

    # step-4: generate prompt
    logging.debug('step-4: generate prompt')
    final_prompt = {
        'question': prompt_template.prompt_generation,
        'docs': lambda x: x['docs'],
    }

    return RunnableParallel(pre_processed) | retrieved | combine_context | final_prompt


# 以下为测试代码
def db_filter(query, db, keywords: dict, embedding_model, mode="no", top_k=2):
    """
        create a filter to improved matchin
        Args:
            query: user query
            db: FAISS db can be a path or vectordb
            keywords: matching template
            embedding_model: embedding model
            mode :[EXACT_MATCH,PART_MATCH,FUZZY_PART_MATCH,FUZZY_EXACT_MATCH,NO_MATCH] default is no match
    """
    print(type(db))
    if isinstance(db, str):
        vectordb = FAISS.load_local(db, embeddings=embedding_model, allow_dangerous_deserialization=True)
    elif isinstance(db, VectorStore):
        # print("!!!!!")
        vectordb = db
    else:
        pass

    if mode == FILTER.EXACT_MATCH:
        res = vectordb.similarity_search(query=query, k=top_k,
                                         filter=lambda x: all(x.get(k) == v for k, v in keywords.items()))
    elif mode == FILTER.FUZZY_EXACT_MATCH:
        res = vectordb.similarity_search(query=query, k=top_k,
                                         filter=lambda x: all(v in x.get(k) for k, v in keywords.items()))
        # res = vectordb.similarity_search(query=query,k=2,filter=lambda d:  "奥特" in d["keywords"])
    elif mode == FILTER.PART_MATCH:
        res = vectordb.similarity_search(query=query, k=top_k,
                                         filter=lambda x: any(x.get(k) == v for k, v in keywords.items()))
    elif mode == FILTER.FUZZY_PART_MATCH:
        res = vectordb.similarity_search(query=query, k=top_k,
                                         filter=lambda x: any(v in x.get(k) for k, v in keywords.items()))
    else:
        res = vectordb.similarity_search(query=query, k=top_k)
    return res


def test_filter(embedding_model):
    d = Document(page_content="text0", metadata={"source": "wiki", "keywords": "奥特兰多"})
    d1 = Document(page_content="text1", metadata={"source": "wiki", "keywords": "奥兰"})
    d2 = Document(page_content="text2", metadata={"source": "wikiii", "keywords": "特兰"})
    d3 = Document(page_content="text3", metadata={"source": "wiki", "keywords": "奥兰"})

    vectordb = FAISS.from_documents([d, d1, d2, d3], embedding=embedding_model)
    score_threshold = 0.0
    keywords = {"source": "wiki", "keywords": "特兰"}
    query = "dadfaf"
    res = db_filter(query=query, db=vectordb, embedding_model=embedding_model, keywords=keywords,
                    mode=FILTER.FUZZY_PART_MATCH, top_k=4)

    print(res)

    # res = vectordb.similarity_search(query=query,k=4,filter=lambda x: any(v in x.get(k) for k, v in c.items()))
    for r in res:
        print(r)


if __name__ == '__main__':
    embedding_model = create_embedding()
    test_filter(embedding_model)