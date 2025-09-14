import matplotlib.pyplot as plt


# 将log中的log.txt文件中的损失曲线画出来
def draw_loss():
    with open('./log/log.txt', 'r') as f:
        lines = f.readlines()
        losses = [float(line.strip().split()[-1]) for line in lines]
    plt.plot(losses)
    plt.show()

draw_loss()