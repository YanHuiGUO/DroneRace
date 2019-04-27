import keras as K
from model import get_DronNet_model
from parse_data import Parse_helper
from generator import TrainImageGenerator, ValGenerator
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,TensorBoard

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
def get_session(gpu_fraction=0.1):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125
class LossHistory(K.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
  
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':

            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

if __name__== '__main__':
    pass
    # generator = TrainImageGenerator(["..\\data\\2019-04-07-12-43-44\\"], batch_size=8,label_size=6)
    # val_generator = ValGenerator("..\\data\\val\\")
        #print (generator.__getitem__())
    # print (val_generator.__getitem__(1))
    nb_epochs = 100
    lr = 0.001
    steps = 1000

    history = LossHistory()
    KTF.set_session(get_session(0.6))  # using 40% of total GPU Memory
    output_path = Path(__file__).resolve().parent.joinpath("checkpoints")
    model = get_DronNet_model(3,lr)
    pre_train_model = 'model.hdf5'
    if os.path.exists(pre_train_model):
        model = K.models.load_model(pre_train_model)
   
    
    generator = TrainImageGenerator(["..\\data\\2019-04-26-19-28-24\\"], batch_size=1,label_size=4)
    val_generator = ValGenerator("..\\data\\eleve_gates_test\\")
    output_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs, lr)),
        ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}.hdf5",
                        monitor="val_loss",
                        verbose=1,
                        mode="auto",
                        save_best_only=False),
    ]

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)
