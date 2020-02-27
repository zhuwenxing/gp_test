import gpkit
from gpkit import Variable, VectorVariable, Model
import numpy as np
import scipy.io as sio

user_num = 3
# h = np.random.rand(user_num, user_num)
data = sio.loadmat("vaild_data_3user.mat")
H = data["abs_H"]
h = np.reshape(H[20], (user_num, user_num))
h = np.transpose(h)
p = VectorVariable(user_num, "p")
var_noise = 1.0
# min_rates = VectorVariable(user_num,"min_rates",[0.5,0.5,0])
min_rates = [0.5,0.5,0] #对于常量而言，使用Variable的意义在哪呢？

p_max = Variable("p_max", 1)
# p_max = 1
constraints = []

items = [] # 目标函数中的f_i(p)/g_i(p)的近似函数，总共有user_num个，目标函数为这些函数的连乘。
p_new = np.random.rand(user_num)  #对功率随机初始化
F = []
G = []
for i in range(user_num):
    s = var_noise
    for j in range(user_num):
        if j != i:
            s = s + h[i, j] ** 2 * p[j]
    signal = h[i, i] ** 2 * p[i]
    inference_plus_noise = s
    inv_sinr = inference_plus_noise / signal
    if min_rates[i]:
        constraints += [(2 ** min_rates[i] - 1) * inv_sinr <= 1]  # bug找到了，如果是min_rate=0的话，就会变成0<=1 ==> True
        # 所以min_rates[i] == 0就不需要将其放在约束中
    constraints += [p[i] / p_max <= 1]
    
    f = inference_plus_noise
    g = f + signal
    F.append(f)
    G.append(g)
count = 0
while True:
    count += 1 # 统计使用了多少次GP
    for i in range(user_num):
        f = F[i]
        g = G[i]
        dict_p = {}
        p_old = p_new
        for i in range(user_num):
            dict_p[p[i]] = p_old[i]  #使用字典的形式构造g.sub()的输入
  
        alpha = [m.sub(dict_p) / g.sub(dict_p) for m in g.chop()]
        # g.chop()可以得到g这个多项式中的每个单项式
        g_tilde = np.prod([(m/alpha[i])**(alpha[i].c) for i,m in enumerate(g.chop())])
        # #上面两行代码为论文中的公式（11）
        
        item = f / g_tilde # i/(1+sinr_i)的近似
        items.append(item)
    objective = np.prod(items)   # 目标函数 
    m = Model(objective, constraints)
    sol = m.solve(verbosity=0) #   求解GP问题
    print(f"第{count}次GP:")
    print("=="*20)
    print(sol.table()) # 打印每次GP的结果
    p_new = sol["variables"][p]  #将计算得到结果提取为array的数据结构，作为下一次的迭代的代入

    if np.linalg.norm((p_new - p_old),ord=1) <= 1e-4:# 退出条件
        break

print(f"The solution is {p_new}")
