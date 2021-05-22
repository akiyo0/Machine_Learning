import math
import logging
import pandas as pd

class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass

class BaseGradientBoosting(AbstractBaseGradientBoosting):
    def __init__(self, loss, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.trees = {}
        self.f_0 = {}
        self.is_log = is_log
        self.is_plot = is_plot

    def fit(self, data): #初始化 f_0(x) 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        self.features = list(data.columns)[1: -1] #掐头去尾， 删除id和label，得到特征名称
        self.f_0 = self.loss.initialize_f_0(data) #对 m = 1, 2, ..., M
        for iter in range(1, self.n_trees+1):
            self.loss.calculate_residual(data, iter)
            target_name = 'res_' + str(iter)
            self.trees[iter] = Tree(data, self.max_depth, self.min_samples_split,
                                    self.features, self.loss, target_name)
            self.loss.update_f_m(data, self.trees, iter, self.learning_rate)
            if self.is_plot:
                plot_tree(self.trees[iter], max_depth=self.max_depth, iter=iter)
        # print(self.trees)
        if self.is_plot:
            plot_all_trees(self.n_trees)
