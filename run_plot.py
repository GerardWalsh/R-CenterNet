import matplotlib.pyplot as plt
import pandas as pd

loss_values = []

with open('train_2.txt', 'r') as ff:
    for i, line in enumerate(ff):
        try:
            # print(line.split(', ')[1].split('Loss:')[-1])
            e = line.split(', ')[0].split(' ')[-1].strip('[').strip(']').split('/')[0]
            # iter = ?
            it = line.split(', ')[1].split(' ')[1].strip('[').strip(']').split('/')[0]
            print(int(it)*int(e))
            loss_values.append([i*30, float(line.split(', ')[1].split('Loss:')[-1])])
        except:
            pass
            # print('::', line)
print(loss_values)
# x = list(range(0, 45*30, 30))
df = pd.DataFrame(loss_values, columns=["iter", "loss"]).set_index('iter')
print(df.head())
df.plot()
plt.show()
# plt.plot(x, loss_values)
# plt.show()