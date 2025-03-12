```bash
python run.py \
    --backend gpt-3.5-turbo \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/tot_10k.log \
    ${@}
```
run.py
```python
def run(args):
    task = HotPotQATask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    count = 0
    task_accs = []
    info = []

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.algorithm == 'lats':
            state, value, all_nodes, reward, em = lats_search(args, task, i, args.iterations, True)
        elif args.algorithm == 'tot':
            state, value, all_nodes, reward, em = dfs_search(args, task, i, args.iterations)
        elif args.algorithm == 'rap':
            state, value, all_nodes, reward, em = mcts_search(args, task, i, args.iterations)
        else:
            raise Exception("Search algorithm option not valid")
         # log main metric
        if em is None:
            em = 0
        task_accs.append(em)
        cnt_avg = sum(task_accs) / len(task_accs)
        print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
    n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))
```
Env

```python
import ast
import json
import time
import gym
import requests
from bs4 import BeautifulSoup

# import wikipedia

def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return p


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

  def search_step(self, entity):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()
    response_text = requests.get(search_url).text
    self.search_time += time.time() - old_time
    self.num_searches += 1
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
    else:
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        self.search_step("[" + entity + "]")
      else:
        self.page = ""
        for p in page:
          if len(p.split(" ")) > 2:
            self.page += clean_str(p)
            if not p.endswith("\n"):
              self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
  
  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    # if self.answer is not None:  # already finished
    #   done = True
    #   return self.obs, reward, done, self._get_info()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }

class HotPotQAWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"
    self.data = json.load(open(data_file))
    self.data = [(d['question'], d['answer']) for d in self.data]
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "hotpot_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      score = (pred == gt)
      return int(score)
    return 0
  
  def get_metrics(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      em = (pred == gt)
      f1 = f1_score(pred, gt)[0]
      return {'reward': em, 'em': em, 'f1': f1}
    return {'reward': 0, 'em': 0, 'f1': 0}

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  
  def __len__(self):
    return len(self.data)
```
task
```python
class HotPotQATask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def test_output(self, idx: int, output: str):
        output = output.split('Action:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model='gpt-4')
        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*correctness score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        # print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y
standard_prompt = '''
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}
'''
```
```python
import itertools
import numpy as np
from functools import partial
from models import gpt
import wikienv, wrappers
import requests
import logging
import random

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="train")
env = wrappers.LoggingWrapper(env)

global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []



class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }          

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))
```

```python
def lats_search(args, task, idx, iterations=30, to_print=True):
    global gpt
    global failed_trajectories
    global reflection_map
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)
    root = Node(state=None, question=x)
    all_nodes = []
    failed_trajectories = []
    terminal_nodes = []
    reflection_map = []
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    for i in range(iterations):
        logging.info(f"Iteration {i + 1}...")
        node = select_node(root)
#----------------------
  def select_node(node):
    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        
        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node  # This will return None if all paths from the root are exhausted
#--------------------------------
        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root)
        
        if node is None:
            logging.info("All paths lead to terminal nodes with reward 0. Ending search.")
            break

        if node.is_terminal and node.reward == 1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            return node.state, node.value, all_nodes, node.reward, node.em
        
        expand_node(node, args, task)

        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            expand_node(node, args, task)
#---------------------------
def expand_node(node, args, task):
    if node.depth >= 7:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample)
    node.children.extend(new_nodes) #여기서 끝 expand_node
#==========================================================
def generate_new_states(node, args, task, n):
    global failed_trajectories
    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample, stop="Observation")
# def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
#     global failed_trajectories
#     global reflection_map
#     unique_trajectories = get_unique_trajectories(failed_trajectories)
#     if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
#         print("generating reflections")
#         reflection_map = task.generate_self_reflection(unique_trajectories, x)
#     if prompt_sample == 'standard':
#         prompt = task.standard_prompt_wrap(x, y)
#     elif prompt_sample == 'cot':
#         prompt = task.cot_prompt_wrap(x, y, reflection_map)
#     else:
#         raise ValueError(f'prompt_sample {prompt_sample} not recognized')
#     logging.info(f"PROMPT: {prompt}")
#     samples = gpt(prompt, n=n_generate_sample, stop=stop)
#     return [y + _ for _ in samples]  
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    tried_actions = []
    
    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        new_state = node.state.copy()  # Make a copy of the parent node's state

        thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
###############
def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      score = (pred == gt)
      return int(score)
    return 0
def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  def white_space_fix(text):
      return " ".join(text.split())
  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)
  def lower(text):
      return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))
#####################
            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done #reward 가 1이거나 done을 step이 return 시 True
            new_node.reward = r
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory(new_node)
# def collect_trajectory(node):
#     trajectory = []
#     while node:
#         trajectory.append(str(node))
#         node = node.parent
#     return '\n'.join(reversed(trajectory))              
                print(trajectory)
              if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list
#================================================ generate_new_states
#---------------------------- expand_nodes
        value = evaluate_node(node, args, task)
        # Find the child with the highest value

#---------------------------------
def evaluate_node(node, args, task):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
#===================================

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values
def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    global reflection_map
    global failed_trajectories
    
    unique_trajectories = get_unique_trajectories(failed_trajectories)
# def get_unique_trajectories(failed_trajectories, num=5):
#     unique_trajectories = []
#     seen_final_answers = set()
#     for traj in failed_trajectories:
#         final_answer = traj.get('final_answer')
#         if final_answer not in seen_final_answers:
#             node_string = traj['trajectory']
#           #node_trajectory_to_text(node_string) 시작
#             lines = node_string.split('\n')
#             formatted_lines = []
#             for line in lines:
#                 try:
#                     depth = int(line.split(",")[0].split("=")[1].strip())
#                     thought = line.split(", thought=")[1].split(", action=")[0].strip()
#                     action = line.split(", action=")[1].split(", observation=")[0].strip()
#                     observation = line.split(", observation=")[1].split(")")[0].strip()
#                 except IndexError:
#                     continue
                
#                 if depth != 0:
#                     if thought:
#                         formatted_lines.append(f"Thought {depth}: {thought}")
#                     if action:
#                         formatted_lines.append(f"Action {depth}: {action}")
#                     if observation:
#                         formatted_lines.append(f"Observation {depth}: {observation}")
            
#             result = '\n'.join(formatted_lines) 
#           #여기까지 node_trajectory_to_text(node_string):
#             unique_trajectories.append(result) 
#             seen_final_answers.add(final_answer)
#         if len(unique_trajectories) >= num:
#             break
#     return unique_trajectories  
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map)
    logging.info(f"Current: {x}")
    logging.info(f"Current: {y}")
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    logging.info(f"VALUE PROMPT: {value_prompt}")
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    logging.info(f"VALUE OUTPUTS: {value_outputs}")
    value = task.value_outputs_unwrap(value_outputs)
    logging.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value
#value_prompt_wrap 함수
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = []) -> str:
        question = x.split('\n')[0]
        #z = []
        if len(z) != 0:
            failed_trajectories = ""
            # Combine the trajectories with their corresponding reflections
            for traj, ref in zip(z, reflections):
                failed_trajectories += f"{question}\n{traj}\nThis trajectory is incorrect as {ref['reflection']}\nThus the correctness score is 1\n"
            inp = x + y + "\nThis trajectory is "
            prompt = value_prompt_reasoning_feedback.format(s="", trajectories=failed_trajectories, input=inp)
            if get_token_length(prompt) > max_token_length:
                prompt = value_prompt_reasoning_feedback_short.format(s="", trajectories=failed_trajectories, input=inp)
        if len(z) != 0 and False:
            failed_trajectories = "\n".join([f"{question}\n{traj}\nThus the correctness score is 1\n" for traj in z])
            inp = x + y + "\nThus the correctness score is "
            prompt = value_prompt_feedback.format(s="", trajectories=failed_trajectories, input=inp)
            if get_token_length(prompt) > max_token_length:
                print("Token length exceeded", get_token_length(prompt))
                failed_trajectories = "\n".join([f"{question}\n{traj}\nThus the correctness score is 1\n" for traj in z[:2]])
                inp = y + "\nThus the correctness score is "
                prompt = value_prompt_feedback.format(s="", trajectories=failed_trajectories, input=inp)
                print("New length", get_token_length(prompt))
        else:
            inp = y + "\nThis trajectory is "
            #inp = y + "\nThus the correctess score is "
            #prompt = value_prompt.format(s="", input=inp)
            prompt = value_prompt_reasoning.format(s="", input=inp)
        return prompt
    def value_outputs_unwrap(evaluate_prompt: str):
        evaluate_prompt = evaluate_prompt[0]
        if '10' in evaluate_prompt:
            return 1.0
        elif '9' in evaluate_prompt:
            return 0.9
        elif '8' in evaluate_prompt:
            return 0.8
        elif '7' in evaluate_prompt:
            return 0.7
        elif '6' in evaluate_prompt:
            return 0.6
        elif '5' in evaluate_prompt:
            return 0.5
        elif '4' in evaluate_prompt:
            return 0.4
        elif '3' in evaluate_prompt:
            return 0.3
        elif '2' in evaluate_prompt:
            return 0.2
        elif '1' in evaluate_prompt:
            return 0.1
        else:
            return -1        
  #==================== get_value 끝
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    for i, child in enumerate(node.children):
        child.value = votes[i] 
    # max_vote = max(votes) if votes else 1
    # if max_vote == 0:
    #     max_vote = 1  # Avoid division by zero
    
    # terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    # for i, condition in enumerate(terminal_conditions):
    #     if condition == 1:
    #         votes[i] = max_vote + 1
    
    # for i, child in enumerate(node.children):
    #     child.value = votes[i] / max_vote  # Now safe from division by zero
    return sum(votes) / len(votes) if votes else 0
#---------------------------------evaluate_node 끝
        reward, terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=4)

#--------------------------------------------
def rollout(node, args, task, idx, max_depth=4):
    logging.info("ROLLING OUT")
    depth = node.depth
    n = 5
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        logging.info(f"ROLLING OUT {depth}")
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n)
   
        for state in new_states:
            if state.is_terminal:
                return state.reward, state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        #new_state = new_state[0]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)

        max_value_index = values.index(max(values))
        rewards.append(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            rewards = [-1]
    
    logging.info("ROLLOUT FINISHED")
    return sum(rewards) / len(rewards), node
#--------------------------------------------------------------
        terminal_nodes.append(terminal_node)

        if terminal_node.reward == 1:
            logging.info("SUCCESSFUL TRAJECTORY FOUND DURING SIMULATION")
            return terminal_node.state, terminal_node.value, [], terminal_node.reward, terminal_node.em

        backpropagate(terminal_node, reward)
#---------------------------------
def backpropagate(node, value):
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                logging.info(f"Backpropagating with reward 0 at depth {node.depth}. New value: {node.value}.")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logging.info(f"Backpropagating with reward 1 at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent
#----------------------------------
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        # Check for terminal nodes with a reward of 1
        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            return best_node.state, best_node.value, all_nodes, best_node.reward, best_node.em
          
    # def collect_all_nodes(node):
    #     """Recursively collect all nodes starting from the given node."""
    #     nodes = [node]
    #     for child in node.children:
    #         nodes.extend(collect_all_nodes(child))
    #     return nodes
        for j, (node, value) in enumerate(all_nodes):
             logging.info(f"Node {j+1}: {str(node)}")
        logging.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")

    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    if best_child.reward == 1:
        logging.info("Successful trajectory found")
    else:
        logging.info("Unsuccessful trajectory found")
    if best_child is None:
        best_child = root
    return best_child.state, best_child.value, all_nodes, best_child.reward, best_child.em
```
prompt
```bash

value_prompt = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]
Thus the correctness score is 3

Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?
Thought 1: I need to search for the football manager who recruited David Beckham, and then find out the timeframe in which he managed Manchester United.
Action 1: Search[David Beckham's football manager]
Observation 1: Could not find David Beckham's football manager. Similar: ['David Beckham', 'Inter Miami CF (redirect from Miami Beckham United)', 'Victoria Beckham', 'The Beckham Experiment', 'Miami Freedom Park'].
Thought 2: I couldn't find the information on David Beckham's football manager. I need to search Manchester United and find out the timeframe of the manager.
Action 2: Search[Manchester United]
Observation 2: Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.. Domestically, Manchester United have won a record 20 league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields.
Thought 3: The information about David Beckham's football manager is not available. I will search for Manchester United's managers to find the timeframe.
Action 3: Search[Manchester United managers]
Observation 3: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thought 4: The information about David Beckham's football manager is not available. The question asks for the timeframe when the manager managed Manchester United, so I need to find a list of Manchester United managers.
Action 4: Search[list of Manchester United managers]
Observation 4: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thus the correctness score is 4

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]
Thus the correctness score is 10

{input}
'''

value_prompt_feedback = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]
Thus the correctness score is 3

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]
Thus the correctness score is 10

{trajectories}
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]
Thus the correctness score is 10

{input}
'''

value_prompt_reasoning = '''You are an advanced reasoning agent that can improve based on self refection. Analyze the trajectories of your previous solutions to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]
This trajectory is incorrect as my search should be related to Mexican Formula One race car drivers, not winners of the Mexican Grand Prix, a seperate event. A better search would have been for the List of Formula One Grand Prix winners, as suggested.
Thus the correctness score is 3

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]
This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.
Thus the correctness score is 10

Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?
Thought 1: I need to search for the football manager who recruited David Beckham, and then find out the timeframe in which he managed Manchester United.
Action 1: Search[David Beckham's football manager]
Observation 1: Could not find David Beckham's football manager. Similar: ['David Beckham', 'Inter Miami CF (redirect from Miami Beckham United)', 'Victoria Beckham', 'The Beckham Experiment', 'Miami Freedom Park'].
Thought 2: I couldn't find the information on David Beckham's football manager. I need to search Manchester United and find out the timeframe of the manager.
Action 2: Search[Manchester United]
Observation 2: Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.. Domestically, Manchester United have won a record 20 league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields.
Thought 3: The information about David Beckham's football manager is not available. I will search for Manchester United's managers to find the timeframe.
Action 3: Search[Manchester United managers]
Observation 3: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thought 4: The information about David Beckham's football manager is not available. The question asks for the timeframe when the manager managed Manchester United, so I need to find a list of Manchester United managers.
Action 4: Search[list of Manchester United managers]
Observation 4: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
This trajectory is mostly incorrect as I was not able to retrieve the relevant information to answer the question. The searches are too broad and not relevant to the specific player recruited, David Beckham. The last two actions are also too similar. It would be better to first search for David Beckham and lookup information related to his manager.
Thus the correctness score is 4

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]
This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search. 
Thus the correctness score is 10

{input}
'''

value_prompt_reasoning_feedback = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]
This trajectory is incorrect as my search should be related to Mexican Formula One race car drivers, not winners of the Mexican Grand Prix, a seperate event. A better search would have been for the List of Formula One Grand Prix winners, as suggested.
Thus the correctness score is 3

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]
This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.
Thus the correctness score is 10

Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?
Thought 1: I need to search for the football manager who recruited David Beckham, and then find out the timeframe in which he managed Manchester United.
Action 1: Search[David Beckham's football manager]
Observation 1: Could not find David Beckham's football manager. Similar: ['David Beckham', 'Inter Miami CF (redirect from Miami Beckham United)', 'Victoria Beckham', 'The Beckham Experiment', 'Miami Freedom Park'].
Thought 2: I couldn't find the information on David Beckham's football manager. I need to search Manchester United and find out the timeframe of the manager.
Action 2: Search[Manchester United]
Observation 2: Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.. Domestically, Manchester United have won a record 20 league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields.
Thought 3: The information about David Beckham's football manager is not available. I will search for Manchester United's managers to find the timeframe.
Action 3: Search[Manchester United managers]
Observation 3: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thought 4: The information about David Beckham's football manager is not available. The question asks for the timeframe when the manager managed Manchester United, so I need to find a list of Manchester United managers.
Action 4: Search[list of Manchester United managers]
Observation 4: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
This trajectory is mostly incorrect as I was not able to retrieve the relevant information to answer the question. The searches are too broad and not relevant to the specific player recruited, David Beckham. The last two actions are also too similar. It would be better to first search for David Beckham and lookup information related to his manager.
Thus the correctness score is 4

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]
This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search. 
Thus the correctness score is 10

{trajectories}

{input}
'''

value_prompt_reasoning_feedback_short = '''Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]
This trajectory is incorrect as my search should be related to Mexican Formula One race car drivers, not winners of the Mexican Grand Prix, a seperate event. A better search would have been for the List of Formula One Grand Prix winners, as suggested.
Thus the correctness score is 3

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]
This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search. 
Thus the correctness score is 10

{trajectories}

{input}
'''
HotPotQA Prompts

E.1

Base Acting Prompt

Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.

(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.

(3) Finish[answer], which returns the answer and finishes the task.

After each observation, provide the next Thought and next Action.

Here are some examples:

Question: Which magazine was started first Arthur’s Magazine or First for Women?

Thought 1: I need to search Arthur’s Magazine and First for Women, and find which was started first.

Action 1: Search[Arthur’s Magazine]

Observation 1: Arthur’s Magazine was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey’s Lady’s Book.[3]

Thought 2: Arthur’s Magazine was started in 1844. I need to search First for Women next.

Action 2: Search[First for Women]

Observation 2: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey.[2] In 2011 the circulation of the magazine was 1,310,696 copies.[3]

Thought 3: First for Women was started in 1989. 1844 (Arthur’s Magazine) 
<
 1989 (First for Women), so Arthur’s Magazine was started first.

Action 3: Finish[Arthur’s Magazine]

(examples)

You have attempted to answer the following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.

(trajectories)

(input)

E.2

Base Reasoning Prompt

Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question. Start your response with either Action or an indexed Thought

Here are some examples:

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Let’s think step by step.

Thought 1: The eastern sector of Colorado orogeny extends into the High Plains.

Thought 2: High Plains rise in elevation from around 1,800 to 7,000 ft

Thought 3: The answer is 1,800 to 7,000 ft.

Action: Finish[1,800 to 7,000 ft]

(examples)

Previous trial: (trajectories)

(input)

E.3Value Function Prompt
Analyze the trajectories of a solution to a question answering task. The trajectories are labeled by environmental Observations about the situation, Thoughts that can reason about the current situation, and Actions that can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.

(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.

(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude “Thus the correctness score is s”, where s is an integer from 1 to 10.

Question: Which magazine was started first Arthur’s Magazine or First for Women?

Thought 1: I need to search Arthur’s Magazine and First for Women, and find which was started first.

Action 1: Search[Arthur’s Magazine]

Observation 1: Arthur’s Magazine was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey’s Lady’s Book.[3]

This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.

Thus the correctness score is 10

(other examples)

(failed trajectories)

(context)

E.4Reflection Prompt
Analyze the trajectories of a solution to a question-answering task. The trajectories are labeled by environmental Observations about the situation, Thoughts that can reason about the current situation, and Actions that can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.

(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.

(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude “Thus the correctness score is s”, where s is an integer from 1 to 10.

Question: Which magazine was started first Arthur’s Magazine or First for Women?

Thought 1: I need to search Arthur’s Magazine and First for Women, and find which was started first.

Action 1: Search[Arthur’s Magazine]

Observation 1: Arthur’s Magazine was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey’s Lady’s Book.[3]

This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.

Thus the correctness score is 10

(other examples)

(failed trajectories)

(context)
```
