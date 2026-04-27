import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorldEnv:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        
        # 设定起点和目标点
        self.start_pos = (0, 0)
        self.target_pos = (9, 9)
        
        # 设定障碍物位置
        self.obstacles = [(2, 2), (2, 3), (2, 4), (5, 5), (5, 6), (5, 7), (7, 2), (7, 3), (8, 8), (2, 7), (3, 7)]
        
        self.action_space = 4 # 0: 上, 1: 下, 2: 左, 3: 右
        self.state_dim = 2
        
        self.current_pos = self.start_pos
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        """重置环境，返回初始状态"""
        self.current_pos = self.start_pos
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """获取归一化后的状态 (x, y)"""
        return np.array([self.current_pos[0] / (self.width - 1), self.current_pos[1] / (self.height - 1)], dtype=np.float32)

    def step(self, action):
        """执行动作，返回 next_state, reward, done, info"""
        self.steps += 1
        x, y = self.current_pos
        
        if action == 0: # 上
            y += 1
        elif action == 1: # 下
            y -= 1
        elif action == 2: # 左
            x -= 1
        elif action == 3: # 右
            x += 1
            
        next_pos = (x, y)
        
        done = False
        reward = -1.0 # 每走一步给一个小惩罚，鼓励更短路径
        info = {'success': False}
        
        # 检查边界
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            reward = -10.0 # 走出边界给予较大惩罚
            done = False
            next_pos = self.current_pos 
        # 检查障碍物
        elif next_pos in self.obstacles:
            reward = -10.0 # 撞到障碍物给予较大惩罚
            done = False
            self.current_pos = next_pos
        # 检查目标
        elif next_pos == self.target_pos:
            reward = 100.0 # 到达目标给予大奖励
            done = True
            info['success'] = True
            self.current_pos = next_pos
        else:
            self.current_pos = next_pos
            
        # 检查最大步数
        if self.steps >= self.max_steps and not done:
            done = True
            
        return self._get_state(), reward, done, info
        
    def render(self, trajectory=None, save_path=None):
        """可视化环境和机器人的移动轨迹"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(self.width + 1):
            ax.axvline(i, color='gray', linestyle='--', linewidth=0.5)
        for i in range(self.height + 1):
            ax.axhline(i, color='gray', linestyle='--', linewidth=0.5)
            
        # 绘制起点和终点
        ax.add_patch(patches.Rectangle((self.start_pos[0], self.start_pos[1]), 1, 1, color='blue', label='起点'))
        ax.add_patch(patches.Rectangle((self.target_pos[0], self.target_pos[1]), 1, 1, color='green', label='目标货架'))
        
        # 绘制障碍物
        for idx, obs in enumerate(self.obstacles):
            label = '障碍物' if idx == 0 else ""
            ax.add_patch(patches.Rectangle((obs[0], obs[1]), 1, 1, color='black', label=label))
            
        # 绘制轨迹
        if trajectory:
            xs = [p[0] + 0.5 for p in trajectory]
            ys = [p[1] + 0.5 for p in trajectory]
            ax.plot(xs, ys, color='red', marker='o', markersize=5, linewidth=2, label='移动轨迹')
            
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # 处理中文字体显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.title('智能仓储机器人路径规划')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
