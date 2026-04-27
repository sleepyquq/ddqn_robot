import numpy as np
import matplotlib.pyplot as plt
from env import GridWorldEnv
from dqn_agent import DQNAgent

def train(num_episodes=1000, render_eval=True):
    env = GridWorldEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space, 
                     lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                     epsilon_decay=0.995, buffer_size=10000, batch_size=64, target_update_freq=10)
    
    rewards_history = []
    success_history = []
    steps_history = []
    
    print(f"开始训练 Double DQN，总回合数 {num_episodes} ...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            
        agent.update_epsilon()
        rewards_history.append(episode_reward)
        success_history.append(1 if info.get('success', False) else 0)
        steps_history.append(env.steps)
        
        if (episode + 1) % 50 == 0:
            success_rate = np.mean(success_history[-50:])
            avg_reward = np.mean(rewards_history[-50:])
            avg_steps = np.mean(steps_history[-50:])
            print(f"episode: {episode+1}/{num_episodes}, Epsilon: {agent.epsilon:.3f}, avg reward: {avg_reward:.2f}, success rate: {success_rate:.2f}, avg steps: {avg_steps:.2f}")
            
    print("训练完成。正在保存训练曲线...")
    plot_training_curves(rewards_history, success_history, steps_history)
    
    import torch
    torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
    print("模型权重已保存至 dqn_model.pth")
    
    if render_eval:
        evaluate(env, agent)

def plot_training_curves(rewards, successes, steps):
    window = 20
    avg_rewards = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
    avg_success = [np.mean(successes[max(0, i-window+1):i+1]) for i in range(len(successes))]
    avg_steps = [np.mean(steps[max(0, i-window+1):i+1]) for i in range(len(steps))]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(rewards, alpha=0.3, color='blue', label='回合奖励')
    axs[0].plot(avg_rewards, color='blue', label=f'移动平均 (窗口={window})')
    axs[0].set_title('训练奖励曲线')
    axs[0].set_ylabel('奖励')
    axs[0].legend()
    
    axs[1].plot(avg_success, color='green', label=f'成功率 (窗口={window})')
    axs[1].set_title('到达目标成功率')
    axs[1].set_ylabel('成功率')
    axs[1].legend()
    
    axs[2].plot(steps, alpha=0.3, color='red', label='回合步数')
    axs[2].plot(avg_steps, color='red', label=f'移动平均 (窗口={window})')
    axs[2].set_title('每回合步数')
    axs[2].set_xlabel('回合')
    axs[2].set_ylabel('步数')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("已生成 training_curves.png")

def evaluate(env, agent):
    print("\n开始评估训练好的策略...")
    state = env.reset()
    done = False
    trajectory = [env.current_pos]
    
    while not done:
        # 评估模式，无探索
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        trajectory.append(env.current_pos)
        
    is_success = info.get('success', False)
    print(f"评估完成。是否到达目标: {is_success}, 消耗步数: {len(trajectory)-1}")
    
    if not is_success:
        print("机器人未能到达目标，可能卡住了或达到了最大步数。")
        
    env.render(trajectory=trajectory, save_path='evaluation_path.png')
    print("已生成 evaluation_path.png")

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
    train()