import numpy
from tensorboardX import SummaryWriter

writer = SummaryWriter()
writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
parameter = {'xsinx': n_iter * np.sin(n_iter), 'xcosx': niter * np.cos(n_iter), 'arctanx': np.ar}
writer.add_scalar('data/scaler_group', {}, n_iter)
