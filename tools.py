import tensorflow as tf


class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,name='relu'):
        with tf.variable_scope(name):
            if name =='relu':
                return  Ops.relu(x)
            else:
                return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            try:
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
            except Exception,e:
                print e

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            in_d = x.get_shape()[1]
            w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
            y = tf.nn.bias_add(tf.matmul(x, w), b)
            Ops.variable_sum(w, name)
            return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name,pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            in_c = x.get_shape()[4]
            w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)

            stride = [1, str, str, str, 1]
            y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
            Ops.variable_sum(w, name)
            return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name,pad='SAME'):
        with tf.variable_scope(name):
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.zeros_initializer()
            bat, in_d1, in_d2, in_d3, in_c = [int(d) for d in x.get_shape()]
            w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
            b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
            out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
            stride = [1, str, str, str, 1]
            y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
            y = tf.nn.bias_add(y, b)
            Ops.variable_sum(w, name)
            return y

    @staticmethod
    def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
        '''Assume 2d [batch, values] tensor'''

        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[-1]
            x_shape = x.get_shape()
            axis = list(range(len(x_shape) - 1))
            scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
            offset = tf.get_variable('offset', [size])

            pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
            pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
            batch_mean, batch_var = tf.nn.moments(x, axis)

            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

            return tf.cond(training, batch_statistics, population_statistics)
