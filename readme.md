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


```python
global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

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

            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
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
#================================================ 
#----------------------------
        value = evaluate_node(node, args, task)
        # Find the child with the highest value

#---------------------------------
def evaluate_node(node, args, task):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
#===================================
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
  #====================    
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
#---------------------------------
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
    #     for j, (node, value) in enumerate(all_nodes):
    #         logging.info(f"Node {j+1}: {str(node)}")
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
