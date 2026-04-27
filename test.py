import torch
import time
import os
from env import GridWorldEnv
from dqn_agent import DQNAgent

def print_grid(env, agent_pos):
    """在终端直观打印当前网格状态"""
    # 清屏 (跨平台)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("Double DQN 智能体自动寻路测试\n")
    print("图例: [R]=机器人(Robot)  S=起点(Start)  T=目标(Target)  X=障碍物(Obstacle)  .=空地\n")
    
    grid_str = ""
    # 从上到下打印 (y 递减)
    for y in range(env.height - 1, -1, -1):
        for x in range(env.width):
            pos = (x, y)
            if pos == agent_pos:
                grid_str += "[R] "
            elif pos == env.start_pos:
                grid_str += " S  "
            elif pos == env.target_pos:
                grid_str += " T  "
            elif pos in env.obstacles:
                grid_str += " X  "
            else:
                grid_str += " .  "
        grid_str += "\n"
    print(grid_str)
    print(f"当前位置: {agent_pos}\n")

def test_model(model_path='dqn_model.pth', render_delay=0.3):
    print(f"Loading model weights from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist. Please run main.py first to train the model.")
        return

    env = GridWorldEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space)
    
    try:
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nStarting visual testing in 2 seconds...")
    time.sleep(2)
    
    state = env.reset()
    done = False
    trajectory = [env.current_pos]
    
    # 初始状态打印
    print_grid(env, env.current_pos)
    time.sleep(render_delay)
    
    while not done:
        # 使用贪心策略 (evaluate=True)
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        trajectory.append(env.current_pos)
        
        # 在终端直观展示
        print_grid(env, env.current_pos)
        time.sleep(render_delay)

    is_success = info.get('success', False)
    print(f"Test finished. Reached target: {is_success}, Steps taken: {len(trajectory)-1}")
    
    env.render(trajectory=trajectory, save_path='test_evaluation_path.png')
    print("Generated final path image: test_evaluation_path.png")

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    test_model()