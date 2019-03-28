from keras.models import Model
from keras.layers import Input, Add, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout,LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation,Flatten
from keras import backend as K
import tensorflow as tf
def get_DronNet_model(input_channel_num = 3):
    
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

    inputs = Input(shape=(240, 320, input_channel_num))
    x = Conv2D(32, (5, 5),strides = 2, padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides = 2)(x)
    
    x = _residual_block(x,32)
	
    x = _residual_block(x,64)
	
    x = _residual_block(x,128)
	
    x = Dropout(0.5)(x)
    #x1 = Dense(1, activation='relu',input_dim = 128)(x)
    #x2 = Dense(4, activation='relu',input_dim = 128)(x)
    
    x = Flatten()(x)
    x = Dense(8)(x)
    x = LeakyReLU(alpha=0.5)(x)
    #x = Dense(64)(x)
    x = Dense(4)(x)
    x = LeakyReLU(alpha=0.5)(x)
    #x = ELU(alpha=1.0)(x)
    # for i in range(resunit_num):
        # x = _residual_block(x)

    # x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    # x = BatchNormalization()(x)
    # x = Add()([x, x0])
    # x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model
if __name__ == "__main__":
    print (get_DronNet_model(3).summary())