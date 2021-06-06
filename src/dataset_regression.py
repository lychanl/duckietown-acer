import models.cnn as cnn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.python.keras import activations


class DataModel(tf.keras.Model):
    def __init__(self, filters: list, kernels: list, strides: list, layers: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = cnn.build_cnn_network(
            filters=filters,
            kernels=kernels,
            strides=strides
        )

        self.hidden_layers = [l for layer in layers for l in [tf.keras.layers.Dense(layer), tf.keras.layers.ReLU()]]

        self.out = tf.keras.layers.Dense(2 + 3 + 2)

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


# @tf.function
def loss(out, y, beta):
    regr_loss = tf.reduce_sum(tf.square(out[:,:2] - y[:,:2]), axis=-1)
    obj_loss = tf.square(out[:,2] - tf.cast(y[:,2] != 0, tf.float32))
    class_loss = tf.square(out[:,3] - tf.cast(y[:,2] == 2, tf.float32))
    obj_regr_loss = tf.reduce_sum(tf.square(out[:,4:] - y[:,3:]), axis=-1)

    obj_flag = tf.cast(y[:,2] == 0, tf.float32)

    return tf.reduce_mean(regr_loss + obj_loss + beta * obj_flag * (class_loss + obj_regr_loss))


def stats(out, y):
    out = out.numpy()
    y = y.numpy()

    regr_loss = np.square(out[:,:2] - y[:,:2]).sum(-1).mean()
    det = out[:,2] > 0.5
    class1 = (out[:,2] > 0.5) & (out[:,3] > 0.5)
    class2 = (out[:,2] > 0.5) & (out[:,3] < 0.5)
    obj_regr_loss = np.square(out[:,4:] - y[:,3:]).sum(-1).mean()

    det_acc = (det == (y[:,2] != 0)).mean()
    det_prec = ((det == 1) & (y[:,2] != 0)).sum() / (y[:,2] != 0).sum()

    c1_acc = ((class1 == (y[:,2] == 1)) * (y[:,2] != 0)).sum() / (y[:,2] != 0).sum()
    c1_prec = ((class1 == 1) & (y[:,2] == 1)).sum() / (y[:,2] == 1).sum()

    c2_acc = ((class2 == (y[:,2] == 2)) * (y[:,2] != 0)).sum() / (y[:,2] != 0).sum()
    c2_prec = ((class2 == 1) & (y[:,2] == 2)).sum() / (y[:,2] == 2).sum()

    return regr_loss, det_acc, det_prec, c1_acc, c1_prec, c2_acc, c2_prec, obj_regr_loss

def train(model, X, Y, lr, beta, steps, batch_size, log=100):
    optimizer = tf.keras.optimizers.Adam(lr)

    train_X = X[:int(X.shape[0] * 0.8)]
    train_Y = Y[:int(X.shape[0] * 0.8)]

    val_X = X[int(X.shape[0] * 0.8):]
    val_Y = Y[int(X.shape[0] * 0.8):]

    losses = 0

    for step in range(steps):
        indices = np.random.randint(train_X.shape[0], size=batch_size)
        x = train_X[indices]
        y = train_Y[indices]

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        with tf.GradientTape() as tape:
            out = model(x)

            l = loss(out, y, beta)

        vars = model.trainable_variables
        grads = tape.gradient(l, vars)

        optimizer.apply_gradients(zip(grads, vars))
        losses += l.numpy()

        if step % log == 0:
            tl = 0
            n = 0
            
            s = np.zeros(8)

            for i in range(val_X.shape[0] // batch_size):
                x = val_X[i*100:(i+1)*100]
                y = val_Y[i*100:(i+1)*100]

                x = tf.convert_to_tensor(x, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.float32)

                out = model(x)

                tl += loss(out, y, beta).numpy()
                n += 1

                s += stats(out, y)

            print(f'Step {step}: train {losses / log}, val {tl / n}')
            regr_loss, det_acc, det_prec, c1_acc, c1_prec, c2_acc, c2_prec, obj_regr_loss = s / n
            print(f'Regr loss {regr_loss} det a {det_acc} p {det_prec} c1 a {c1_acc} p {c1_prec} c2 a {c2_acc} p {c2_prec} Obj regr {obj_regr_loss}')
            losses = 0


if __name__ == '__main__':
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    model = DataModel(
        filters=[8, 16, 16, 32, 32],
        kernels=[1, 5, 3, 3, 1],
        strides=[1, 4, 2, 1, 1],
        layers=[64, 32, 32]
    )
    
    X = np.load(r"D:\projects\duckietown-acer\MultiMap-v0_dataset_x.npy")
    y = np.load(r"D:\projects\duckietown-acer\MultiMap-v0_dataset_y.npy")

    X = X / 256

    print('training...')

    train(model, X, y, 0.0002, 3, 500000, 200, 500)

    model.save_weights('data_model2/weights')

    pass
