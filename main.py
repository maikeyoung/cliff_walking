# -*- codeing = utf-8 -*-
# @Time : 2020/10/25 21:27
# @Author : xiao
# @File : main.py
# @Software: PyCharm

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
'''
代码的想法，创建一个Q表，储存每个动作对应的48个状态的价值
首先创建一个空的Q表，Q表伟二维数组，四组，每组48个值，分别对应整个世界中的48个状态
这里将48定为一个智能体的绝对位置，智能体的x，y轴形式定为相对位置
对q-learning算法的分解：1、行为策略为ε-greedy算法，这里传入智能体的位置和Q表来算出此时应该选择的动作
                      2、确定执行这个动作后所得奖励与下一个状态
                      3、更新Q表的函数：传入q表和，学习效率，衰减率，现在的位置，现在所选择的动作，下一个状态值，
'''

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from maze import Maze

world_height = 4
world_long = 12

ALPHA = 0.5
EPSILON = 0.1
GAMMA = 1

action_up = 0
action_down = 1
action_left = 2
action_right = 3
actions = [action_up,action_down,action_left,action_right]

start = [3,0]
goal = [3,11]


'''
   获得的参数：  state：现在所取的状态
                q_table：已更新的Q值表
                EPSILON = 0.1
'''
def epilon_greedy(state,q_table,episode):             #根据现在的状态，在q_table中选取应该采取的行动
    decide_exploration = np.random.random()  # 在0~1之间选取一个随机值
    global EPSILON
    if (episode > 20000):
        EPSILON = 0.01
    if (episode >=49999):
        EPSILON = 0.1
    if (decide_exploration < EPSILON):
        action = np.random.choice(4)  # 0：up，1：down，2：left 3：right    从0~3中随机选取一个动作作为下一次动作
    else:
        action = np.argmax(q_table[:, state])               #:代表在该二维数组中前面还有一个值   缺省值需要表示出来  两个数组用，隔开
    return action

'''
确定动作后移动智能体
输入的参数：agent：智能体的位置
          action:智能体的下一个动作
输出：智能体下一个位置
'''
def step(agent,action):               #智能体行动，给入现在的智能体位置以及未来的行动
    (X,Y)=agent
    #UP
    if ((action == 0) and (X > 0) ):
        X -= 1
    #down
    elif ((action == 1) and (X < 3)):
        X += 1
    #left
    elif ((action == 2) and (Y > 0)):
        Y -= 1
    #RIGHT
    elif ((action == 3) and (Y < 11)):
        Y += 1

    agent = (X,Y)
    return agent

'''
得到智能体位置来得到状态价值
得到最大的价值

输入：智能体位置
     q表
输出：智能体的动作价值
     智能体绝对位置
'''

def getstate(agent,q_table):
    (X,Y) = agent
    state = 12*X + Y            #确定了现在的绝对位置
    agent_action = q_table[: ,state]
    MAX_actionvalue = np.amax(agent_action)
    return state,MAX_actionvalue

'''
得到未来的位置会给予的奖励

输入：未来位置
输出：结束游戏的标记以及reward
'''
def getreward(state):#state  ->   [0,47]
    reward = -1
    END = False
    if state == 47:
        reward = 1
        END = True
    if state < 47 and state > 36:
        reward = -100
        END = True
    return  reward,END


def up_qtable(state,action,next_actionvalue,q_table,reward):
    q_table[action,state] =  q_table[action,state] + ALPHA * (reward+GAMMA*next_actionvalue  - q_table[action,state])

    return q_table
''''
    如果没有到目标就不结束循环，到了悬崖也不会终止循环,这里迭代500次
    为了之后的画图，这里需要储存每一次迭代所需要的收益和迭代数
    q-learning算法
    输入：迭代的次数
    输出：q表，奖励的累积，步骤的累积
'''
def q_learning(episodes):
    reward_cache = list()
    step_cache = list()
    q_table = np.zeros((4,48))
    for episode in range(0,episodes):
        if episode > episodes-100:
            env.reset()
        END = False
        reward_count = 0    #奖励的积累
        episode_count = 0
        step_count =0
        state = 36
        agent = start
        while END == False:   #agent != goal
            action = epilon_greedy(state,q_table,episode)       #贪心算法确定下一步的动作
            if episode > episodes-100:
                env.render()
                env.step(action)
            agent = step(agent,action)                  #更新智能体的下一个相对位置
            step_count += 1                          #统计智能体行走的步数
            next_state,max_value = getstate(agent,q_table) #得到下一个位置和最大动作价值
            reward,END = getreward(next_state)                #到下一个位置应该得到的奖励与状态
            reward_count += reward                       #累加奖励
            q_table = up_qtable(state,action,max_value,q_table,reward) #更新q表
            state = next_state
        reward_cache.append(reward_count)

        if episode >= (episodes-1):
            print("q-learning已经迭代%d次" %(episodes))
        step_cache.append(step_count)
    env.destroy()
    return q_table,reward_cache,step_cache


''''
    如果没有到目标就不结束循环，到了悬崖也不会终止循环,这里迭代500次
    为了之后的画图，这里需要储存每一次迭代所需要的收益和迭代数
    sarsa算法
    输入：迭代的次数
    输出：q表，奖励的累积，步骤的累积
'''

def sarsa(episodes):
    reward_cache = list()
    step_cache = list()
    q_table = np.zeros((4,48))
    for episode in range(0, episodes):
        if episode > episodes - 100:
            env.reset()
        END = False
        reward_count = 0  # 奖励的积累
        episode_count = 0
        step_count = 0
        agent = start
        state = 36
        action = epilon_greedy(state,q_table,episode)
        while END == False:
            # agent != goal
            agent = step(agent, action)  # 更新智能体的下一个相对位置
            if episode > episodes - 100:
                env.render()
                env.step(action)
            step_count += 1  # 统计智能体行走的步数
            next_state,_ = getstate(agent,q_table)
            reward, END = getreward(next_state)  # 到下一个位置应该得到的奖励与状态
            reward_count += reward  # 累加奖励
            next_action = epilon_greedy(next_state, q_table,episode)
            next_value = q_table[next_action][next_state]
            q_table = up_qtable(state, action, next_value, q_table, reward)  # 更新q表
            state = next_state
            action = next_action
        reward_cache.append(reward_count)

        if episode >= (episodes-1):
            print("SARSA已经迭代%d次" %(episodes))
        step_cache.append(step_count)
    env.destroy()
    return q_table, reward_cache, step_cache
'''
    function：对数据集中的n个值取平均值
    input：取平均值的数据集
           
    output：返回一个去玩平均值的数集

'''
def average(reward,n):
    fun_average = list(np.zeros(int(len(reward) / n)))
    for a in range(int(len(reward) / n)):
        for i in range(10):
            fun_average[a] = fun_average[a] + reward[a * n + i]
    for i in range(len(fun_average)):
        fun_average[i] = fun_average[i] / n
    return fun_average
'''
对q-learning与sarsa算法的测试集
输入得到的各幕数奖励值，画曲线做对比
'''
def cum_reward(reward_qlearing,reward_sarsa):

   # reward_q = average(reward_qlearing,10)
    #reward_s = average(reward_sarsa,10)
    reward_q = reward_qlearing[::100]
    reward_s = reward_sarsa[::100]
    plt.figure(dpi=80)
    plt.ylim(-150,0)
    plt.plot(reward_q,label = 'q-learning')
    plt.plot(reward_s,label='sarsa')
    plt.xlim(0,500)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['left'].set_position(('data', 0))
    plt.legend(loc = 'best',ncol = 2)
    plt.show()

'''
    输入：q表
    输出：最佳动作策略的表格形式

'''
def the_best_action(q_table):
    optimal_policy = []
    for i in range(0,world_height):
        optimal_policy.append([])
        for j in range(0,world_long):
            if [i,j] == goal:
                optimal_policy[-1].append('G')
                continue
            best_action = np.argmax(q_table[:,i*12+j])
            if best_action == 0:
                optimal_policy[-1].append('U')
            elif best_action == 1:
                optimal_policy[-1].append('D')
            elif best_action == 2:
                optimal_policy[-1].append('L')
            elif best_action == 3:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)

if __name__ == '__main__':
    env = Maze()
    q_table_1,q_learning_cache,step_qlearning_cache = q_learning(50000)
    env = Maze()
    q_table_2,q_sarsa_cache,step_sarsa_cache = sarsa(50000)
    #print(q_table_1)
    the_best_action(q_table_1)
    print('------------------------------------------')
    #print(q_table_2)
    the_best_action(q_table_2)
    # print(q_learning_cache)
    # print(q_sarsa_cache)
    # print(step_sarsa_cache)
    cum_reward(q_learning_cache,q_sarsa_cache)
    env.mainloop()