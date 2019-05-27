import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

init_kernel = tf.contrib.layers.xavier_initializer()

def network(x_input, is_training=False, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('network', reuse=reuse):
        kernel_dense = tf.get_variable('kernel_dense', [120, 128], initializer=init_kernel)
        bias_dense = tf.get_variable('bias_dense', [128])
        kernel_dense2 = tf.get_variable('kernel_dense2', [128, 512], initializer=init_kernel)
        bias_dense2 = tf.get_variable('bias_dense2', [512])
        bias_inv_dense2 = tf.get_variable('bias_inv_dense2', [128])
        bias_inv_dense = tf.get_variable('bias_inv_dense', [120])

        x = tf.nn.softplus(tf.matmul(x_input, kernel_dense) + bias_dense)
        x = tf.nn.softplus(tf.matmul(x, kernel_dense2) + bias_dense2)

        ## inverse layers
        x = tf.nn.softplus(tf.matmul(x, tf.transpose(kernel_dense2)) + bias_inv_dense2)
        x = tf.nn.softplus(tf.matmul(x, tf.transpose(kernel_dense)) + bias_inv_dense)

    return x

class DSEBM():
    def __init__(self, opts):
        self.config = opts
        self.x_input = tf.placeholder(tf.float32, shape=[None, 120], name='input')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        noise = tf.random_normal(shape=tf.shape(self.x_input), mean=0.0, stddev=1., dtype=tf.float32)
        self.x_noise = self.x_input + noise

        b_prime = tf.get_variable('b_prime', shape=[opts['batch_size'], 120])
        self.net_out = network(self.x_input, self.is_training)
        self.net_nosie_out = network(self.x_noise, self.is_training)

        self.energy = 0.5 * tf.reduce_sum(tf.square(self.x_input - b_prime)) - tf.reduce_sum(self.net_out)
        self.energy_noise = 0.5 * tf.reduce_sum(tf.square(self.x_noise - b_prime)) - tf.reduce_sum(self.net_nosie_out)

        fx = self.x_input - tf.gradients(self.energy, self.x_input)
        self.fx = tf.squeeze(fx, axis=0)
        self.fx_noise = self.x_noise - tf.gradients(self.energy_noise, self.x_noise)

        self.loss = tf.reduce_mean(tf.square(self.x_input - self.fx_noise))

        ## energy score
        flat = tf.layers.flatten(self.x_input - b_prime)
        self.list_score_energy = 0.5 * tf.reduce_sum(tf.square(flat), axis=1) - tf.reduce_sum(self.net_out, axis=1)

        ## recon score
        delta = self.x_input - self.fx
        delta_flat = tf.layers.flatten(delta)
        self.list_score_recon = tf.norm(delta_flat, ord=2, axis=1, keep_dims=False)

        self.add_optimizers()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_optimizers(self):
        opts = self.config
        opt = tf.train.AdamOptimizer(learning_rate=opts['lr'])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_net = [x for x in update_ops if ('network' in x.name)]

        with tf.control_dependencies(update_ops_net):
            self.opt = opt.minimize(self.loss)

    def train(self, data):
        opts = self.config
        batch_size = opts['batch_size']
        num_points = data.train_data.shape[0]
        batch_num = num_points // batch_size

        for epoch in range(opts['epoch_num']):
            sum_loss = 0
            sum_energy = 0
            sum_energy_noise = 0
            for ii in range(batch_num):
                batch_index = np.random.choice(num_points, batch_size, replace=False)
                batch_data = data.train_data[batch_index]

                feed_d = {
                    self.x_input: batch_data,
                    self.is_training: True,
                }
                [_, loss, energy, energy_noise] = self.sess.run([self.opt, self.loss, self.energy, self.energy_noise],\
                                                                feed_dict=feed_d)
                sum_loss += loss
                sum_energy += energy
                sum_energy_noise += energy_noise
            print("Epoch %d, Loss %g, energy %g, energy noise %g" % (epoch, sum_loss/batch_num, \
                                                sum_energy/batch_num, sum_energy_noise/batch_num))
            self.eval(data)

    def eval(self, data):
        opts = self.config
        num_test_points = data.test_data.shape[0]
        batch_size = opts['batch_size']
        batch_num = num_test_points//batch_size
        energy_score = np.zeros((1, ))
        recon_score = np.zeros((1, ))
        true_label = np.zeros((1,1))
        for ii in range(batch_num):
            batch_index = np.random.choice(num_test_points, batch_size, replace=False)
            batch_data = data.test_data[batch_index]
            batch_label = data.test_label[batch_index]
            feed_d = {
                self.x_input: batch_data,
                self.is_training: False,
            }
            [score_e, score_r] = self.sess.run([self.list_score_energy, self.list_score_recon], feed_dict=feed_d)
            energy_score = np.concatenate([energy_score, score_e])
            recon_score = np.concatenate([recon_score, score_r])
            true_label = np.concatenate([true_label, batch_label], axis=0)
        energy_score = energy_score[1:]
        recon_score = recon_score[1:]
        true_label = true_label[1:]
        print("DSEBM-e:")
        self.compute_score(energy_score, true_label)
        print("DSEBM-r:")
        self.compute_score(recon_score, true_label)

    def compute_score(self, score_list, labels):
        num_test_points = labels.shape[0]
        score_sort_index = np.argsort(score_list)
        y_pred = np.zeros_like(labels)
        y_pred[score_sort_index[-int(num_test_points * 0.2):]] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(labels.astype(int),
                                                                   y_pred.astype(int),
                                                                   average='binary')
        print("precision: %g, recall: %g, f1: %g" % (precision, recall, f1))