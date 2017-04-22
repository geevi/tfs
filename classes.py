from builtins import object


class BaseModel(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def train(self, sess):
        return sess.run([self.optimizer, self.train_summary_op, self.global_step])[1:]

    def validate(self, sess):
        return sess.run(self.val_summary_op)
