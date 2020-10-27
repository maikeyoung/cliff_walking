# -*- codeing = utf-8 -*-
# @Time : 2020/10/26 9:21
# @Author : xiao
# @File : maze.py
# @Software: PyCharm
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT =40
MAZE_H = 4
MAZE_W = 12

class Maze(tk.Tk,object):                                #从Tk类中创建子类Maze
    def __init__(self):
        super(Maze,self).__init__()                      #从父类中调用函数来重载
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))       #设置窗口尺寸
        self._build_maze()
    def _build_maze(self):
        self.canvas = tk.Canvas(self,bg = 'white',
                                height = MAZE_H*UNIT,
                                width = MAZE_W*UNIT)

        #创建栅格
        for c in range(0,MAZE_W*UNIT,UNIT):
            x0,y0,x1,y1 = c,0,c,MAZE_H * UNIT
            self.canvas.create_line(x0,y0,x1,y1)
        for r in range(0,MAZE_H*UNIT,UNIT):
            x0,y0,x1,y1 = 0,r,MAZE_W*UNIT,r
            self.canvas.create_line(x0, y0, x1, y1)


        origin = np.array([20,20])
        hell1_center = origin + np.array([UNIT * 1, UNIT * 3])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        hell2_center = origin + np.array([UNIT * 2, UNIT * 3])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        hell3_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        hell4_center = origin + np.array([UNIT * 4, UNIT * 3])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
        hell5_center = origin + np.array([UNIT * 5, UNIT * 3])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')
        hell6_center = origin + np.array([UNIT * 6, UNIT * 3])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')
        hell7_center = origin + np.array([UNIT * 7, UNIT * 3])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')
        hell8_center = origin + np.array([UNIT * 8, UNIT * 3])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')
        hell9_center = origin + np.array([UNIT * 9, UNIT * 3])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - 15, hell9_center[1] - 15,
            hell9_center[0] + 15, hell9_center[1] + 15,
            fill='black')
        hell10_center = origin + np.array([UNIT * 10, UNIT * 3])
        self.hell10 = self.canvas.create_rectangle(
            hell10_center[0] - 15, hell10_center[1] - 15,
            hell10_center[0] + 15, hell10_center[1] + 15,
            fill='black')
        oval_center = origin + np.array([UNIT * 11, UNIT * 3])  # 宝藏
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        agent_center = origin + np.array([UNIT * 0,UNIT * 3])    # 智能体
        self.agent= self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

        self.canvas.pack()  # 打包

    def reset(self):                        #重新开始一幕重新设定agent的位置
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)  # 配置Python Tkinter 画布（Canvas），删除变化的矩形，然后重新创建
        origin = np.array([20, 20])
        agent_center = origin + np.array([UNIT * 0, UNIT * 3])  # 智能体
        self. agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')# 创建中心位置红色方块代表当前位置

    def step(self, action):
        s = self.canvas.coords(self.agent)                      #得到现在的位置
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

    def render(self):
        time.sleep(0.1)
        self.update()


# def update():
#     for t in range(10):
#         env.step(0)
#         env.step(0)


if __name__ == '__main__':
    env = Maze()
    env.step(0)
    env.step(0)
    env.step(3)
    time.sleep(1)
    env.reset()
    env.mainloop()
