import argparse

import models.cnn as cnn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, initializers
from tensorflow.python.keras import activations


class DataModel(tf.keras.Model):
    def __init__(self, filters: list, kernels: list, strides: list, layers: list, *args, l1: float=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = cnn.build_cnn_network(
            filters=filters,
            kernels=kernels,
            strides=strides,
            initializer=initializers.glorot_uniform
        )

        for layer in self.cnn_layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.L1(l1)

        self.hidden_layers = [l for layer in layers for l in [
            tf.keras.layers.Dense(
                layer, kernel_regularizer=tf.keras.regularizers.L1(l1),
                kernel_initializer=initializers.glorot_uniform),
            tf.keras.layers.ReLU()
        ]]

        self.out = tf.keras.layers.Dense(
            2 + 3 + 2, kernel_regularizer=tf.keras.regularizers.L1(l1),
            kernel_initializer=initializers.glorot_uniform)

    def cnn_forward(self, x):
        out = x
        for layer in self.cnn_layers:
            out = layer(out)

        return out

    def __call__(self, x):
        out = self.cnn_forward(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.out(out)
        
        dist = out[:,0]
        angle = out[:,1]
        det = tf.sigmoid(out[:,2])
        clss = tf.sigmoid(out[:,3])
        obj_dist = out[:,4]
        obj_angle = out[:,5]

        return tf.stack([dist, angle, det, clss, obj_dist, obj_angle], axis=1)


def log_loss(y, p):
    p = tf.clip_by_value(p, 0.001, 0.999)
    return -(y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))


# @tf.function
def loss(out, y, beta, eta=0):
    regr_loss = tf.reduce_sum(tf.square(out[:,:2] - y[:,:2]), axis=-1)
    obj_loss = log_loss(tf.cast(y[:,2] != 0, tf.float32), out[:,2])
    class_loss = log_loss(tf.cast(y[:,2] == 2, tf.float32), out[:,3])
    obj_regr_loss = tf.square(out[:,4] - 1 / (1 + y[:,3])) + tf.square(out[:,5] - y[:,4])

    obj_weight = tf.where(y[:,2] == 0, 0, 1 / (1 + y[:,3]))

    return tf.reduce_mean(regr_loss + obj_loss + beta * obj_weight * (class_loss + obj_regr_loss))


def stats(out, y):
    out = out.numpy()
    y = y.numpy()

    regr_loss = np.square(out[:,:2] - y[:,:2]).sum(-1).mean()
    det = out[:,2] > 0.5
    class1 = (out[:,2] > 0.5) & (out[:,3] < 0.5)
    class2 = (out[:,2] > 0.5) & (out[:,3] > 0.5)
    obj_regr_loss = np.square(out[:,4] - 1 / (1 + y[:,3])).mean() + np.square(out[:,5] - y[:,4]).mean()

    det_acc = (det == (y[:,2] != 0)).mean()
    det_prec = ((det == 1) & (y[:,2] != 0)).sum() / (y[:,2] != 0).sum()

    c1_acc = ((class1 == (y[:,2] == 1)) * (y[:,2] != 0)).sum() / (y[:,2] != 0).sum()
    c1_prec = ((class1 == 1) & (y[:,2] == 1)).sum() / (y[:,2] == 1).sum()

    c2_acc = ((class2 == (y[:,2] == 2)) * (y[:,2] != 0)).sum() / (y[:,2] != 0).sum()
    c2_prec = ((class2 == 1) & (y[:,2] == 2)).sum() / (y[:,2] == 2).sum()

    return regr_loss, det_acc, det_prec, c1_acc, c1_prec, c2_acc, c2_prec, obj_regr_loss

def eval_model(model, X, Y, beta, batch_size, train_part):
    tl = 0
    n = 0
    
    s = np.zeros(8)

    for i in range(int(X.shape[0] * train_part), X.shape[0], batch_size):
        x = X[i:(i+batch_size)] / 256
        y = Y[i:(i+batch_size)]

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        out = model(x)

        tl += loss(out, y, beta).numpy()
        n += 1

        s += stats(out, y)

    return tl / n, s / n

def train(model, X, Y, lr, beta, steps, batch_size, train_part=0.8, log=100):
    optimizer = tf.keras.optimizers.Adam(lr)

    losses = 0
    best_loss = np.inf

    for step in range(steps):
        indices = np.random.randint(int(X.shape[0] * train_part), size=batch_size)
        x = X[indices] / 256
        y = Y[indices]

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            out = model(x)

            l = loss(out, y, beta)
            reg = tf.reduce_sum(model.losses)
            tl = l + reg

        vars = model.trainable_variables
        grads = tape.gradient(tl, vars)

        optimizer.apply_gradients(zip(grads, vars))
        losses += l.numpy()

        if step % log == 0:
            eval_q, stats = eval_model(model, X, Y, beta, batch_size, train_part)
            if eval_q < best_loss:
                best_loss = eval_q
                best_step = step
                weights = [w.numpy() for w in model.weights]

            print(f'Step {step}: train {losses / log}, val {eval_q}')
            regr_loss, det_acc, det_prec, c1_acc, c1_prec, c2_acc, c2_prec, obj_regr_loss = stats
            print(f'Regr loss {regr_loss} det a {det_acc} p {det_prec} c1 a {c1_acc} p {c1_prec} c2 a {c2_acc} p {c2_prec} Obj regr {obj_regr_loss}')
            losses = 0

        
    eval_q, stats = eval_model(model, X, Y, beta, batch_size, train_part)
    if eval_q < best_loss:
        best_loss = eval_q
        best_step = step
        weights = [w.numpy() for w in model.weights]

    print(f'Step {steps}: train {losses / log}, val {eval_q}')
    regr_loss, det_acc, det_prec, c1_acc, c1_prec, c2_acc, c2_prec, obj_regr_loss = stats
    print(f'Regr loss {regr_loss} det a {det_acc} p {det_prec} c1 a {c1_acc} p {c1_prec} c2 a {c2_acc} p {c2_prec} Obj regr {obj_regr_loss}')
    losses = 0

    print(f"Best val loss {best_loss} at step {best_step}")
    for w, v in zip(model.weights, weights):
        w.assign(v)


if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--filters', type=int, nargs='+', default=[16, 32, 32, 64, 64])
    parser.add_argument('--kernels', type=int, nargs='+', default=[1, 5, 3, 3, 1])
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 4, 2, 1, 1])
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128])

    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=4)
    parser.add_argument('--batch_size', type=int, default=200)

    parser.add_argument('--steps', type=int, default=int(5e5))
    parser.add_argument('--train_part', type=float, default=0.9)
    parser.add_argument('--log_interval', type=int, default=int(2000))

    parser.add_argument('--in_', type=str, default=None)
    parser.add_argument('--out', type=str, default='data_model')

    parser.add_argument('--X', type=str, default='MultiMap-v0_dataset_x_1M.npy')
    parser.add_argument('--Y', type=str, default='MultiMap-v0_dataset_y_1M.npy')

    args = parser.parse_args()

    model = DataModel(
        filters=args.filters,
        kernels=args.kernels,
        strides=args.strides,
        layers=args.layers,
        l1=args.l1
    )

    X = np.load(args.X)
    y = np.load(args.Y)

    if args.in_:
        model(X[:1])  # initialize model
        model.load_weights(f'{args.in_}/weights')

    print('training...')

    try:
        train(model, X, y, args.lr, args.beta, args.steps, args.batch_size, args.train_part, args.log_interval)
    except KeyboardInterrupt:
        i = None
        while i not in ('y', 'n'):
            i = input('Save model? [y/n]').lower()
        
        if i == 'n':
            exit(0)

    model.save_weights(f'{args.out}/weights')
