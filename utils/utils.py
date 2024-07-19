import json
import logging
import os
import time
import warnings
import torch
from typing import List
from operator import itemgetter
from deprecated.sphinx import deprecated
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from . import prompt_template, splitter

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.vectorstores")


def merge_vectordbs(vectordbs: List[str], target_name: str, target_root: str = '~/.cache/rag/vectordb'):
    """
        Merge several vector databases
        :param vectordbs: list of paths
        :param target_name: name of the merged vectordb
        :param target_root: save_path
        :return: merged vectordb
    """
    embedding_model = create_embedding()
    target_db = os.path.expanduser(os.path.join(target_root, target_name))

    # base
    base = FAISS.load_local(vectordbs[0], embeddings=embedding_model, allow_dangerous_deserialization=True)
    logging.info(f"load vectordb {vectordbs[0]}, size={base.index.ntotal}")
    for vectordb in vectordbs[1:]:
        add = FAISS.load_local(vectordb, embeddings=embedding_model, allow_dangerous_deserialization=True)
        logging.info(f"load vectordb {vectordb}, size={add.index.ntotal}")
        base.merge_from(add)
    base.save_local(target_db)
    logging.info("---- merge vectordb into {} | size={} ----".format(target_db, base.index.ntotal))


@deprecated(reason='use merge_vectordbs instead', version='1.0')
def merge_two_vector_db(origin_db: str or VectorStore, add_db: str or VectorStore, save_path: str):
    """
        Merge two vector databases
        :param origin_db: path / VectorStore
        :param add_db: path / VectorStore
        :param save_path
        :return:
    """
    embedding_model = create_embedding()
    new_db_path = os.path.expanduser(save_path)

    # original
    if isinstance(origin_db, str):
        origin_db = os.path.expanduser(origin_db)
        original = FAISS.load_local(origin_db, embeddings=embedding_model, allow_dangerous_deserialization=True)
    elif isinstance(origin_db, VectorStore):
        original = origin_db
    else:
        logging.error("Original db type error")
        return
    logging.info("---- Finish loading original db ----")

    # add
    if isinstance(add_db, str):
        add_db = os.path.expanduser(add_db)
        add = FAISS.load_local(add_db, embeddings=embedding_model, allow_dangerous_deserialization=True)
    elif isinstance(add_db, VectorStore):
        add = add_db
    else:
        logging.error("Add db type error")
        return
    logging.info("---- Finish loading add db ----")

    # merge
    logging.info("merge two db with size: {} and {}".format(original.index.ntotal, add.index.ntotal))
    original.merge_from(add)
    original.save_local(new_db_path)
    logging.info("---- Finish merging | size={} ----".format(original.index.ntotal))


def create_vector_db(doc_root: str, db_name: str, db_root="~/.cache/rag/classdb",
                     split_fmt: str = 'manual', multi_index: bool = True, expand_size: int = 2048,
                     page_num: bool = False, keywords: bool = True, overwrite: bool = True):
    """
        Creating a vector database from file paths
        Args:
            :param doc_root: file root (no sub-dir)
            :param db_name: name of the vectordb (save -> db_root/db_name/manual_chunk/xx.faiss)
            :param db_root: save_path
            :param split_fmt: manual / chunk
            :param multi_index: copy docs and replace page_content with summary
            :param expand_size: target chunk length of chunk expand
            :param page_num: used for load_split_manual
            :param keywords: add keywords into metadata
            :param overwrite: overwrite the vectordb if db_root/db_name is not empty
    """
    # fixme: add {'content_type'='original'} in metadata
    db_root = os.path.expanduser(os.path.join(db_root, db_name))
    embedding_model = create_embedding()

    # check overwrite (if not empty)
    if os.path.isdir(db_root) and len(os.listdir(db_root)) > 0:
        if overwrite:
            logging.warning(f"{db_root} is not empty, will overwrite")
        else:
            logging.error(f"{db_root} is not empty, please set overwrite=True")
            return

    # load documents
    if split_fmt == 'manual':
        docs = splitter.load_split_manual(doc_root, page_num=page_num)
    elif split_fmt == 'chunk':
        docs = splitter.load_split_chunk(doc_root)
    else:
        logging.error("split_fmt error")
        return

    # expand and add {'content_type': 'original'}
    docs = splitter.chunk_expand(docs, expand_size)
    for doc in docs:
        doc.metadata.update({'content_type': 'original'})

    # add keywords in metadata
    if keywords:
        logging.info("add keywords into metadata")
        docs = splitter.chunk_update_keywords(docs, create_model(0.5))

    # save NAME.expNk.faiss
    name = f'{split_fmt}.exp{expand_size//1024}k.faiss'
    logging.info(f'creating vectorstore: {name}')
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    vectorstore.save_local(os.path.join(db_root, name))
    logging.info("save complete: {}".format(os.path.join(db_root, name)))

    # save NAME.expNk.multi.faiss
    if multi_index:
        name = f'{split_fmt}.exp{expand_size // 1024}k.multi.faiss'
        multi_doc = splitter.chunk_multi_index(docs, model=create_model(0.5), use_expanded_content=True)
        logging.info(f'creating vectorstore: {db_root}/{name}')
        vectorstore = FAISS.from_documents(multi_doc, embedding=embedding_model)
        vectorstore.save_local(os.path.join(db_root, name))
        logging.info("save complete: {}".format(os.path.join(db_root, name)))


def clear_db(db_name: str, db_path: str, embedding_model):
    logging.info("---- Start Empty Vector Database ----")
    vectorstore = FAISS.load_local(folder_path=os.path.join(db_path, db_name), embeddings=embedding_model,
                                   allow_dangerous_deserialization=True)
    vectorstore.delete(list(vectorstore.index_to_docstore_id.values()))
    logging.info("---- Empty Vector Database Over ----")


def update_db(ids, doc, db_path, embedding_model, db_name=None):
    vectorstore = FAISS.load_local(folder_path=os.path.join(db_path, db_name), embeddings=embedding_model,
                                   allow_dangerous_deserialization=True)
    vectorstore.delete([ids])
    vectorstore.add_texts(texts=[doc.page_content], metadatas=[doc.metadata], ids=ids)


def add_docs(db_path: str, docs: list[Document], embedding_model, db_name=None) -> None:
    """
        Adding files to an existing vector database
        Args:
            db_path: Faiss database path
            db_name: Faiss database name
            docs: uploaded documents
            embedding_model: embedding model
    """
    if db_name is None:
        folder_path = db_path
    else:
        folder_path = os.path.join(db_path, db_name)
    vectorstore = FAISS.load_local(folder_path=folder_path, embeddings=embedding_model,
                                   allow_dangerous_deserialization=True)
    logging.info("Before add:{}".format(vectorstore.index.ntotal))
    for doc in docs:
        vectorstore.add_texts(texts=[doc.page_content], metadatas=[doc.metadata])
    vectorstore.save_local(folder_path=db_path)
    logging.info("After add:{}".format(vectorstore.index.ntotal))


def get_pdf_url(filename):
    files = json.load(open('../documents/metadata/file_url.json', 'r'))
    if filename in files.keys():
        return files[filename]
    else:
        return 'http://www.tju.edu.cn/'


def get_chain_note(chain_name):
    chain_notes = {
        'pre_prompt_chain_manual': '【PIPELINE】preprocess + m split + metadata',
        'pre_prompt_chain_manual_expand': '【PIPELINE】preprocess + m split (expand) + metadata',
        'pre_prompt_chain_manual_multi': '【PIPELINE】preprocess + m split (multi-index) + metadata',
        'prompt_chain_manual': '【PIPELINE】m split + metadata',
        'prompt_chain_manual_expand': '【PIPELINE】m split (expand) + metadata',
        'prompt_chain_manual_multi': '【PIPELINE】m split (multi-index) + metadata',
        'prompt_chain_manual_expand_multi': '【PIPELINE】m split (expand w/ multi index) + metadata',
        'prompt_chain_chunk': '【PIPELINE】chunk split',
        'prompt_chain_chunk_expand': '【PIPELINE】chunk split (expand)',
        'prompt_chain_chunk_multi': '【PIPELINE】chunk split (multi-index)',
        'prompt_chain_chunk_expand_multi': '【PIPELINE】chunk split (expand w/ multi index)',
        'prompt_chain_sep': '【PIPELINE】seperator split',
        'prompt_chain_sep_expand': '【PIPELINE】seperator split (expand)',
        'prompt_chain_sep_multi': '【PIPELINE】seperator split (multi-index)',
        'prompt_chain_sep_expand_multi': '【PIPELINE】seperator split (expand w/ multi index)',
        'prompt_chain_page': '【PIPELINE】page split',
        'prompt_chain_page_expand': '【PIPELINE】page split (expand)',
        'prompt_chain_page_multi': '【PIPELINE】page split (multi-index)',
        'prompt_chain_page_expand_multi': '【PIPELINE】page split (expand w/ multi index)',
        'prompt_chain_gpt': '【PIPELINE】gpt-3.5 split',
        'prompt_chain_gpt_expand': '【PIPELINE】gpt-3.5 split (expand)',
        'prompt_chain_gpt_multi': '【PIPELINE】gpt-3.5 split (multi-index)',
        'prompt_chain_gpt_expand_multi': '【PIPELINE】gpt-3.5 split (expand w/ multi index)',
    }
    return chain_notes[chain_name]


def create_model(temperature: float, streaming: bool = False):
    return ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://10.10.111.43:8000/v1",
        temperature=temperature,
        model_name="Qwen1.5-72B-Chat",
        streaming=streaming,
    )


def create_embedding(model_name='infgrad/stella-large-zh-v3-1792d'):
    # DMetaSoul/Dmeta-embedding         1024 tokens
    # infgrad/stella-large-zh-v3-1792d  512  tokens
    # infgrad/stella-base-zh-v3-1792d   1024 tokens
    # NOTE: for more embedding models, ref: https://huggingface.co/spaces/mteb/leaderboard
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    logging.info('created embedding model: {}'.format(model_name))
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def create_retriever(_embedding, _docs: List[Document] = None, _db_path: str = None, score_threshold=0.4):
    # NOTE: for more vectorstore, ref: https://python.langchain.com/docs/integrations/providers
    # todo: support top-k retrieval & filter (search_kwargs)
    if _db_path and _docs:
        logging.error('both _docs and _db_path are provided, please provide only one')
        raise ValueError
    if not _docs and not _db_path:
        logging.error('please provide either _docs or _db_path')
        raise ValueError

    if _docs:
        vectorstore = FAISS.from_documents(_docs, embedding=_embedding)
        logging.info('built vectorstore with {} documents'.format(len(_docs)))
    else:
        vectorstore = FAISS.load_local(_db_path, embeddings=_embedding, allow_dangerous_deserialization=True)
        logging.info('loaded vectorstore from {}'.format(_db_path))
    _retriever = vectorstore.as_retriever(
        search_type='similarity_score_threshold',  # mmr, similarity, similarity_score_threshold
        search_kwargs={'score_threshold': score_threshold, 'k': 8},
    )
    return _retriever


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
        'question': itemgetter('question'),  # RunnablePassthrough()
        'ret_question': itemgetter('question')  # RunnablePassthrough()
    }

    # step-2: load and split
    logging.debug('step-2: create retriever')
    db_root = os.path.expanduser('~/.cache/rag/vectordb')
    retriever = create_retriever(_embedding=embedding_model, _db_path=os.path.join(db_root, cls + '.faiss'))
    retrieved = {
        'docs': itemgetter('ret_question') | retriever,
        'question': itemgetter('question')  # RunnablePassthrough()
    }

    # step-3: combine context
    logging.debug('step-3: combine context')
    combine_context = {
        'context': lambda x: combine_docs(x['docs'], expand=expand),
        'question': itemgetter('question'),
        'docs': itemgetter('docs'),
    }

    # step-4: generate prompt
    logging.debug('step-4: generate prompt')
    final_prompt = {
        'question': prompt_template.prompt_generation,
        'docs': lambda x: x['docs'],
    }

    logging.info(f'finish creating prompt chain: {cls}. time = {time.time() - _start:.2f} sec')

    return RunnableParallel(pre_processed) | retrieved | combine_context | final_prompt


def combine_docs(_docs: List[Document], doc_sep='', expand=False, show_wiki_add=True):
    # combine top-k similar documents into a single string
    # _doc_prompt = ChatPromptTemplate.from_template(template="{page_content}")
    if not _docs:
        return '没有相关资料'
    meta_key = 'metadata' if 'metadata' in _docs[0].metadata.keys() else 'source'

    combined = ''
    for i, d in enumerate(_docs):
        if not show_wiki_add and (d.metadata[meta_key].startswith('wiki-') or d.metadata[meta_key].startswith('add-')):
            continue
        combined += f'【相关资料{i}】**' + d.metadata[meta_key] + '**：'
        # content: expanded page_content / page_content / metadata['page_content']
        if expand and 'expand' in d.metadata.keys():
            combined += d.metadata['expand']
        elif 'content_type' in d.metadata.keys() and d.metadata['content_type'] == 'summary':
            combined += d.metadata['page_content']
        else:
            combined += d.page_content
        combined += doc_sep
    return combined


def print_dir(dir='../'):
    print(os.listdir(dir))


if __name__ == '__main__':
    get_pdf_url('test')
