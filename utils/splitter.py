import concurrent.futures
import copy
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import prompt_template
from typing import List
from bisect import bisect_left

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def split_string_avg(text1, chunk_size):  # 用于搞定大于chunk_size的单句
    # developed by Junhao Li
    i = 1
    while True:
        i += 1
        if len(text1) // i + (len(text1) % i == 0) <= chunk_size:
            break

    chunk_s = len(text1) // i
    chunk_l = len(text1) % i

    chunks = []
    for j in range(i):
        start = j * chunk_s + min(j, chunk_l)
        end = (j + 1) * chunk_s + min(j + 1, chunk_l)
        chunks.append(text1[start:end])
    return chunks


def split_by_single(texts, sign='。', chunks=1, overlap=0):
    # developed by Junhao Li
    # texts:document类型的整页列表, sign:切分标识, chunks:连续出现~个sign就切分
    logging.debug('start split by single seperator')  # fixme: modified by zzx

    all_of = []  # 切分结果
    src = texts[0].metadata['source']  # 来源

    pagec = ""  # 长字符串
    page_positions = []  # 用于记录每页内容在大字符串中的起始位置

    # 整合成长字符串，并且记录页的起始位置
    for text in texts:
        page_positions.append(len(pagec))
        pagec += text.page_content

    # 查找所有匹配项位置
    sign_pos = []
    for i in range(len(pagec)):
        if pagec[i] == sign:
            sign_pos.append(i)

    # 以防文章不以句号结尾
    if sign_pos == [] or sign_pos[-1] != len(pagec) - 1:
        sign_pos.append(len(pagec) - 1)

    # 切分
    start_ind = 0
    chunk_now = 0  # 记录当前chunk数
    overlap_ind = 0
    for i, end_ind in enumerate(sign_pos):
        chunk_now += 1
        if chunk_now == chunks or end_ind == sign_pos[-1]:  # 若chunks数累计达成或已经到达文章结尾就进行一次切分
            chunk_now = 0
            cut_sentense = pagec[start_ind:end_ind + 1]  # 切分的句子
            if overlap:
                overlap_ind = sign_pos[min(i + overlap, len(sign_pos) - 1)]  # fixme: modified by zzx
                cut_sentense += pagec[end_ind + 1, overlap_ind + 1]
            pagenow = bisect_left(page_positions, start_ind)  # 二分查找所在页码
            start_ind = end_ind + 1  # 更新起始位置
            all_of.append(Document(page_content=cut_sentense,
                                   metadata={'source': src, 'page': pagenow, 'expand': cut_sentense}))
    return all_of


def split_single_near_chunks(texts, sign='。', chunk_size=200, overlap=1, overlap_rate=0.2):
    # developed by Junhao Li
    # texts:document类型的整页列表, sign:切分标识, chunk_size:切分句的字符数不超过~
    logging.debug('start split single near chunks')  # fixme: modified by zzx

    all_of = []  # 切分结果
    src = texts[0].metadata['source']  # 来源

    # 先进行句的切分
    texts = split_by_single(texts, sign, chunks=1, overlap=0)

    cut_sentense = ""  # 切分的句子
    start_ind = 0
    # 组合分句
    i = 0
    while i < len(texts):
        if len(cut_sentense) + len(texts[i].page_content) <= chunk_size:  # 若增加后不超过chunk_size，那么可增加
            cut_sentense += texts[i].page_content
            i += 1
        else:
            if start_ind != i:  # 一般情况，此时将进行一次切分,并且开始overlap计算
                overlap_len = 0
                overlap_cnt = 0
                all_of.append(Document(page_content=cut_sentense,
                                       metadata={'source': src, 'page': texts[start_ind].metadata['page'],
                                                 'expand': cut_sentense, 'content_type': 'original'}))
                while i > start_ind + 1 and overlap_cnt < overlap and float(
                        overlap_len + len(texts[i - 1].page_content)) / float(len(cut_sentense)):  # 回退i
                    overlap_cnt += 1
                    overlap_len += len(texts[i - 1].page_content)
                    i -= 1
                cut_sentense = ""
                start_ind = i
            else:  # 处理一个句子过长的特殊情况
                split_chunks = split_string_avg(texts[i].page_content, chunk_size)
                for split_chunk in split_chunks:
                    all_of.append(Document(page_content=split_chunk,
                                           metadata={'source': src, 'page': texts[start_ind].metadata['page'],
                                                     'expand': split_chunk, 'content_type': 'original'}))
                start_ind += 1
                i += 1
    if cut_sentense != "":
        all_of.append(Document(page_content=cut_sentense,
                               metadata={'source': src, 'page': texts[start_ind].metadata['page'],
                                         'expand': cut_sentense, 'content_type': 'original'}))
    return all_of


def chunk_expand(docs: List[Document], chunk_size: int = 2048) -> List[Document]:
    logging.info('start chunk expand')
    _start = time.time()
    # copy docs to avoid modifying the original list
    docs = copy.deepcopy(docs)
    for i, doc in tqdm(enumerate(docs)):
        expand_content = doc.page_content
        for j in range(i + 1, len(docs)):
            expanded = False
            # expand forward first
            if docs[j].metadata['source'] == doc.metadata['source']:
                if len(expand_content) + len(docs[j].page_content) > chunk_size:
                    break
                expand_content += docs[j].page_content
                expanded = True
            # expand backward
            if i - (j - i) >= 0 and docs[i - (j - i)].metadata['source'] == doc.metadata['source']:
                if len(expand_content) + len(docs[i - (j - i)].page_content) > chunk_size:
                    break
                expand_content = docs[i - (j - i)].page_content + expand_content
                expanded = True
            # if no expansion, break
            if not expanded:
                break
        # update metadata
        doc.metadata.update({'expand': expand_content})
    logging.info('chunk expand finished, time = {:.2f} sec'.format(time.time() - _start))
    return docs


def chunk_update_keywords(docs: List[Document], model, use_expanded_content: bool = False,
                          threads: int = 20) -> List[Document]:
    def process_doc(_doc: Document, _model, _use_exp):
        for _ in range(5):
            output = model.invoke(prompt_template.prompt_extract_keywords.format_messages(
                context=_doc.page_content if not use_expanded_content else doc.metadata['expand']))
            if output.content.startswith('keywords'):
                _doc.metadata.update({'keywords': output.content.split('@')[1]})
                return _doc
            else:
                logging.warning(f'[RETRY] failed to extract keywords and summary from {output.content}.')
        _doc.metadata.update({'keywords': ''})
        return _doc

    logging.info('start chunk update keywords')
    logging.warning('temperature of the model should >> 0.0')
    _start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for doc in docs:
            futures.append(executor.submit(process_doc, doc, model, use_expanded_content))
        updated_docs = []
        for future in tqdm(concurrent.futures.as_completed(futures)):
            updated_docs.append(future.result())

    logging.info('chunk update keywords finished, time = {:.2f} sec'.format(time.time() - _start))
    return updated_docs


def chunk_multi_index(docs: List[Document], model, use_expanded_content: bool = False,
                      threads: int = 20) -> List[Document]:
    def process_doc(_doc: Document, _model, _use_exp):
        for _ in range(5):
            output = model.invoke(prompt_template.prompt_extract_summary.format_messages(
                context=_doc.page_content if not use_expanded_content else doc.metadata['expand']))
            if output.content.startswith('summary'):
                _sum_doc = copy.deepcopy(_doc)
                _sum_doc.metadata.update({'content_type': 'summary', 'page_content': _doc.page_content})
                _sum_doc.page_content = output.content.split('@')[1]
                return [_doc, _sum_doc]
            else:
                logging.warning(f'[RETRY] failed to extract keywords and summary from {output.content}.')
        return [_doc]

    logging.info('start chunk multi index')
    logging.warning('temperature of the model should >> 0.0')
    _start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for doc in docs:
            futures.append(executor.submit(process_doc, doc, model, use_expanded_content))
        multi_index_docs = []
        for future in tqdm(concurrent.futures.as_completed(futures)):
            multi_index_docs.extend(future.result())

    logging.info('chunk multi index finished, time = {:.2f} sec'.format(time.time() - _start))
    return multi_index_docs


def load_split_chunk(pdf_root, sign='。', chunk_size=200, overlap=1, overlap_rate=0.2):
    logging.info('start load and split by chunk')
    all_docs = []
    pdf_paths = []
    if os.path.exists(pdf_root):
        for entry in os.listdir(pdf_root):
            if entry.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(pdf_root, entry))
        if not pdf_paths:
            logging.error(f"no pdf file found in {pdf_root}")
            return []
    else:
        logging.error(f"The specified directory {pdf_root} does not exist.")
        return []

    for pdf_path in tqdm(pdf_paths):
        all_texts = []
        loader = PyPDFLoader(file_path=pdf_path)
        pdf_docs = loader.load()
        for _ in pdf_docs:
            _.page_content = re.sub(r'[\n\s]+|-\d+-', '', _.page_content)
        all_texts.extend(pdf_docs)
        all_texts = split_single_near_chunks(all_texts, sign, chunk_size, overlap, overlap_rate)
        all_docs.extend(all_texts)
    logging.info('load and split by chunk finished')
    return all_docs


def load_split_manual(txt_root: str, chunk_size=512, page_num=False) -> List[Document]:
    # fixme: save Docs into save_path
    logging.info('start loading manual split docs...')
    spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 3)
    docs = []
    for file in tqdm(os.listdir(txt_root)):
        source = file.split('；')[0]
        metadata = ''.join(file.split('.')[:-1])
        metadata = metadata.split('/')[-1]
        metadata = metadata.split('\\')[-1]
        contexts = open(os.path.join(txt_root, file), 'r').read().split('\n\n')
        page_nums = []
        if page_num:  # read the first line as page_num (manual split)
            for i in range(len(contexts)):
                page_nums.append(int(contexts[i].split('\n')[0]))
                contexts[i] = '\n'.join(contexts[i].split('\n')[1:])
        i = 0
        while i < len(contexts):
            context = contexts[i]
            # small chunks: merge until chunk_size
            if len(context) < chunk_size:
                while len(context) < chunk_size and i + 1 < len(contexts):
                    if len(context + contexts[i + 1]) > chunk_size:
                        break
                    context += contexts[i + 1]
                    i += 1
                context = context.replace('\n', '')
                if len(context) == 0:  # skip empty context
                    i += 1
                    continue
                if page_num:
                    docs.append(Document(page_content=context, metadata={
                        'source': source, 'metadata': metadata, 'content_type': 'original', 'page_num': page_nums[i]}))
                else:
                    docs.append(Document(page_content=context, metadata={
                        'source': source, 'metadata': metadata, 'content_type': 'original'}))
            # large chunks: split by chunk_size
            else:
                context = context.replace('\n', '')
                for chunk in spliter.split_text(context):
                    if page_num:
                        docs.append(Document(page_content=chunk, metadata={
                            'source': source, 'metadata': metadata, 'content_type': 'original', 'page_num': page_nums[i]}))
                    else:
                        docs.append(Document(page_content=chunk, metadata={
                            'source': source, 'metadata': metadata, 'content_type': 'original'}))
            i += 1
    logging.info('loaded {} manual split docs'.format(len(docs)))
    return docs

