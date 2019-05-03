from keras.models import Model
from keras.layers import Input, Add, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout,LeakyReLU,GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation,Flatten
from keras import regularizers
from keras import backend as K
import tensorflow as tf
from keras.optimizers import Adam
def get_DronNet_model(input_channel_num = 3, lr = 0.001):
    
    def _residual_block(inputs,feature_dim):
        x_0= Conv2D(feature_dim, (1, 1),strides = 2, padding="same", kernel_initializer="he_normal")(inputs)
        x = Conv2D(feature_dim, (3, 3),strides = 2, padding="same", kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal",)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        m = Add()([x, x_0])

        return m

    inputs = Input(shape=(200, 200, input_channel_num))
    x = Conv2D(32, (5, 5),strides = 2, padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides = 2)(x)
    
    x = _residual_block(x,32)
	
    x = _residual_block(x,64)
	
    x = _residual_block(x,128)
	
    x = Dropout(0.5)(x)
    
    features = Flatten()(x)
    
    L2_norm = 0.1
    
    x_1 = Dense(4,kernel_regularizer=regularizers.l2(L2_norm))(features)
    x_1 = LeakyReLU(alpha=0.5)(x_1)
    x_1 = Dense(1,kernel_regularizer=regularizers.l2(L2_norm))(x_1)
    x_1 = LeakyReLU(alpha=0.5,name = 'r')(x_1)
    
    
    x_2 = Dense(4,kernel_regularizer=regularizers.l2(L2_norm))(features)
    x_2 = LeakyReLU(alpha=0.5)(x_2)
    x_2 = Dense(1,kernel_regularizer=regularizers.l2(L2_norm))(x_2)
    x_2 = LeakyReLU(alpha=0.5,name = 'theta')(x_2)
    
    x_3 = Dense(4,kernel_regularizer=regularizers.l2(L2_norm))(features)
    x_3 = LeakyReLU(alpha=0.5)(x_3)
    x_3 = Dense(1,kernel_regularizer=regularizers.l2(L2_norm))(x_3)
    x_3 = LeakyReLU(alpha=0.5, name = 'phi')(x_3)
    
    x_4 = Dense(4,kernel_regularizer=regularizers.l2(L2_norm))(features)
    x_4 = LeakyReLU(alpha=0.5)(x_4)
    x_4 = Dense(1,kernel_regularizer=regularizers.l2(L2_norm))(x_4)
    x_4 = LeakyReLU(alpha=0.5,name = 'yaw')(x_4)
    

    loss_type = "mse"
    model = Model(inputs=inputs, outputs=[x_1,x_2,x_3,x_4])
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss=loss_type, metrics=['mae'],loss_weights={\
                  'r': 1,\
                  'theta': 1,\
                  'phi': 1,\
                  'yaw': 10})
    return model
if __name__ == "__main__":
    print (get_DronNet_model(3).summary())