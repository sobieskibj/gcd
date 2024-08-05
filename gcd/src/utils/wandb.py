import wandb

def log_wandb_scatter(x_ax_data, y_ax_data, x_ax_name, y_ax_name, id):
    data = [[x, y] for x, y in zip(x_ax_data, y_ax_data)]
    table = wandb.Table(data = data, columns = [x_ax_name, y_ax_name])
    wandb.log({id: wandb.plot.scatter(table, x_ax_name, y_ax_name)})
