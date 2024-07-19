import logging
from typing import Dict, Tuple

from . import prompt_template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json


from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://10.10.111.43:8000/v1",
    temperature=0.5,  # note: set temperature to 0.5 for non-deterministic results
    model_name="Qwen1.5-72B-Chat",
)


class DFA:
    def __init__(self, words):
        self.words = words
        self.build()

    def build(self):
        self.transitions = {}
        self.fails = {}
        self.outputs = {}
        state = 0
        for word in self.words:
            current_state = 0
            for char in word:
                next_state = self.transitions.get((current_state, char), None)
                if next_state is None:
                    state += 1
                    self.transitions[(current_state, char)] = state
                    current_state = state
                else:
                    current_state = next_state
            self.outputs[current_state] = word
        queue = []
        for (start_state, char), next_state in self.transitions.items():
            if start_state == 0:
                queue.append(next_state)
                self.fails[next_state] = 0
        while queue:
            r_state = queue.pop(0)
            for (state, char), next_state in self.transitions.items():
                if state == r_state:
                    queue.append(next_state)
                    fail_state = self.fails[state]
                    while (fail_state, char) not in self.transitions and fail_state != 0:
                        fail_state = self.fails[fail_state]
                    self.fails[next_state] = self.transitions.get((fail_state, char), 0)
                    if self.fails[next_state] in self.outputs:
                        self.outputs[next_state] += ', ' + self.outputs[self.fails[next_state]]

    def search(self, text):
        state = 0
        result = []
        for i, char in enumerate(text):
            while (state, char) not in self.transitions and state != 0:
                state = self.fails[state]
            state = self.transitions.get((state, char), 0)
            if state in self.outputs:
                result.append((i - len(self.outputs[state]) + 1, i))
        return result


def filter_words(text, words):
    dfa = DFA(words)
    result = []
    flag = False
    for start_index, end_index in dfa.search(text):
        result.append((start_index, end_index))
    if len(result) != 0:
        # print(result)
        flag = True
    for start_index, end_index in result[::-1]:
        text = text[:start_index] + '*' * (end_index - start_index + 1) + text[end_index + 1:]
    return flag, text


def get_illegal_words():
    words = ['小熊维尼']
    return words


def judge(question):
    flag, res = filter_words(question, get_illegal_words())
    if flag:
        print("检测到违禁词")
        return False
    return True


def format_output_str(output: str) -> Dict or None:  # deprecated
    try:
        outputs = output.split("\n")
        if len(outputs) == 1 and outputs[0].startswith('action'):
            return {'action': outputs[0].split(":")[1].strip()}
        if len(outputs) == 3:
            formatted = {
                'action': eval(outputs[0].split(":")[1].strip()),
                'keywords': outputs[1].split(":")[1].strip().replace('，', ',').split(','),
                'query': outputs[2].split(":")[1].strip()
            }
            print(formatted)
            print(type(formatted['action']))
            return formatted
    except Exception as e:
        pass
    logging.warning("output format wrong, retrying... OUTPUT: [{}]".format(output.replace("\n", "|")))
    return None


def format_output_json(output: str) -> Dict or None:
    formatted = output.replace(" ", "").replace("\n", "")
    try:
        formatted: Dict = json.loads(formatted)
        # check if the json is formatted correctly
        if 'action' not in formatted.keys() or not isinstance(formatted['action'], bool):
            raise KeyError
        if formatted['action'] and ('keywords' not in formatted.keys() or 'query' not in formatted.keys()):
            raise KeyError
        # split keywords by ','
        if formatted['action']:
            formatted['keywords'] = formatted['keywords'].replace('，', ',').split(',')
        else:
            formatted['keywords'] = None
            formatted['query'] = None
        return formatted
    except (json.JSONDecodeError, KeyError):
        logging.warning("output format wrong, retrying... OUTPUT: [{}]".format(output.replace("\n", "\\n")))
        return None


def preprocess(question, max_retry: int = 3) -> Dict or None:
    # question_file_path = "./question.txt"
    prompt = prompt_template.prompt_preprocess_action.format(question=question)

    for i in range(max_retry):
        output = model.invoke(input=prompt).content
        formatted = format_output_json(output)
        if formatted:
            return formatted
    return None


if __name__ == '__main__':
    # print(json.loads('{"action":"True","keywords":"[研究生,出国,政策]","query":"天津大学的研究生出国政策是什么？"}'))
    print(preprocess("怎么考天大"))
    print(preprocess("who are you"))

# question_file_path = "./question.txt"
# def get_question():
#     question_list = []
#     with open(question_file_path,'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             question_list.append(line.rstrip("\n"))

#     return question_list

# print(template)
# question_list = get_question()
# _prompt = PromptTemplate.from_template(template)


# ans_dic_list = []
# for q in question_list:
#     print("question:{}".format(q))
#     prompt = _prompt.format(question = q)    
#     ans = model.invoke(input = prompt).content
#     ans = ans.replace(" ", "").replace("\n", "")
#     ans_json = json.loads(ans)
#     ans_json["action"] = json.loads(ans_json["action"])
#     ans_dic_list.append(ans_json)


# for q in ans_dic_list:
#     print(type(q["action"]))
