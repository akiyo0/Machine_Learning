from visdom import Visdom

viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([loss.item()], [global_step], win="train_loss", update="append")