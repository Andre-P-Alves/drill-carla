import pandas as pd
import matplotlib.pyplot as plt

eval_df = pd.read_json('eval_var.json')
no_train_df = pd.read_json('no_train_var.json')
print(eval_df['acc_var'].mean())
print(no_train_df['acc_var'].mean())
print(eval_df['steer_var'].mean())
print(no_train_df['steer_var'].mean())
plt.plot(range(len(eval_df['acc_var'])), eval_df['acc_var'], label='eval')
plt.plot(range(len(no_train_df['acc_var'])), no_train_df['acc_var'], label='no_train')
plt.savefig(f'plot.png')