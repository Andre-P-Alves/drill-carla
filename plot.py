import pandas as pd
import matplotlib.pyplot as plt

eval_df = pd.read_json('eval_var.json')
no_train_df = pd.read_json('no_train_var.json')
print(eval_df['acc_var'].mean())
print(no_train_df['acc_var'].mean())
print(eval_df['steer_var'].mean())
print(no_train_df['steer_var'].mean())

data = [eval_df['acc_var'].mean(), no_train_df['acc_var'].mean(), eval_df['steer_var'].mean(), no_train_df['steer_var'].mean()]

file_name = "medias.txt"

# Use 'with' statement to open and automatically close the file
with open(file_name, "w") as file:
    # Write each line of data to the file
    for line in data:
        file.write(str(line) + "\n")

plt.xlim(0, 2000)  
plt.xlabel('Estados')
plt.ylabel('Variância das ações')
plt.plot(range(len(eval_df['acc_var'])), eval_df['acc_var'], label='eval')
plt.plot(range(len(no_train_df['acc_var'])), no_train_df['acc_var'], label='no_train')
plt.legend(labels=['Demonstração conhecida', 'Demonstração desconhecida'], loc='upper right')
plt.savefig(f'plot_acc.png')
plt.savefig('plot_acc.eps', format='eps', dpi=600)

plt.figure()

plt.xlim(0, 2000)  
plt.xlabel('Estados')
plt.ylabel('Variância das ações')
plt.plot(range(len(eval_df['steer_var'])), eval_df['steer_var'], label='eval')
plt.plot(range(len(no_train_df['steer_var'])), no_train_df['steer_var'], label='no_train')
plt.legend(labels=['Demonstração conhecida', 'Demonstração desconhecida'], loc='upper right')
plt.savefig(f'plot_steer.png')
plt.savefig('plot_steer.eps', format='eps', dpi=600)

df = pd.read_json(f'actions_var{1}.json')

# Assuming you have the shape of the original 'acc_action' as (2000, 5)
original_shape = (2194, 5)

# Extract the 'acc_action' column and convert it back to a NumPy array
acc_action_list = df['acc_action'].values.tolist()
acc_action = np.array(acc_action_list)
print(acc_action.shape)
# Verify if the shape matches the original shape
if acc_action.shape == original_shape:
    print("Data has been successfully reshaped back to its original shape.")
else:
    print("Data does not match the original shape.")
