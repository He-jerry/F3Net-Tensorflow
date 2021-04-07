from keras.models import *
from bilinear_upsampling import BilinearUpsampling
import tensorflow as tf
from keras import backend as K

from attention import *
import keras
class BatchNorm(BatchNormalization):
    def call(self, inputs, training=None):
          return super(self.__class__, self).call(inputs, training=True)
def BN(input_tensor):
    bn = BatchNorm()(input_tensor)
    a = Activation('relu')(bn)
    return a


#keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest", **kwargs)
def basicblocks(input,filter,dilates=1):
    x1=Conv2D(filter, (3, 3), padding='same',dilation_rate=1*dilates)(input)
    x1=BN(x1)
    #x1=Activation('relu',name='block1_relu')
    return x1



#F3Net
def bottleneck(inputs,inplanes,planes,stride=1,downsample=None,dilation=1):
    x1=Conv2D(planes,kernel_size=(1,1),use_bias=False,padding='same')(inputs)
    x1=BN(x1)
    x2=Conv2D(planes,kernel_size=(3,3),strides=stride,padding='same',use_bias=False,dilation_rate=dilation)(x1)
    x2=BN(x2)
    x3=Conv2D(planes*4,kernel_size=(1,1),use_bias=False,padding='same')(x2)
    x3=BatchNorm()(x3)
    cc=Concatenate(axis=-1)([x3,inputs])
    x4=Activation('relu')(cc)
    return x4

def bottlechain(input,inplanes,plane,blocks,stride=1,dilation=1):
    x1=bottleneck(inputs=input,inplanes=64,planes=plane,stride=1,dilation=1)
    for i in range(1,blocks):
        x1=bottleneck(inputs=x1,inplanes=64*4,planes=plane,dilation=dilation)
    return x1

def makelayer(inputs,planes,blocks,stride=1,dilation=1):
    x1=Conv2D(planes*4,kernel_size=(1,1),strides=stride,use_bias=False,padding='same')(inputs)
    x1=BatchNorm()(x1)
    x1=bottlechain(x1,64,planes,blocks,stride,dilation)
    return x1
def resize_like(input_tensor,ref_tensor,name=None):
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    size=(H.value,W.value)
    if name is None:
      return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))(input_tensor)
    else:
      return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True),name=name)(input_tensor)
def ResNet(inputs):
    conv1=Conv2D(64,kernel_size=7,strides=2,padding='same',use_bias=False)(inputs)
    conv1=BN(conv1)
    conv1=keras.layers.MaxPool2D((3,3),strides=2)(conv1)

    conv2=makelayer(conv1,64,3,1,1)
    conv3=makelayer(conv2,128,4,2,1)
    conv4=makelayer(conv3,256,6,2,1)
    conv5=makelayer(conv4,512,3,2,1)

    return conv2,conv3,conv4,conv5
def basicblock(inputs):
    c1=Conv2D(64,kernel_size=3,strides=1,padding='same')(inputs)
    c1=BatchNorm()(c1)
    c1=Activation('relu')(c1)
    return c1

def cfm(left,down):
    #resize
    left=resize_like(left,down)

    #blocks
    out1h=basicblock(left)
    out2h=basicblock(out1h)
    out1v=basicblock(down)
    out2v=basicblock(out1v)

    out2h=resize_like(out2h,out2v)

    fuse=keras.layers.multiply([out2h,out2v])
    bfuse=basicblock(fuse)
    bfuse=resize_like(bfuse,out1h)
    out3h=keras.layers.add([bfuse,out1h])
    out4h=basicblock(out3h)
    bfuse = resize_like(bfuse, out1v)
    out3v=keras.layers.add([basicblock(bfuse),out1v])
    out4v=basicblock(out3v)
    return out4h,out4v
def decoder(out2h,out3h,out4h,out5v):
    out4h,out4v=cfm(out4h,out5v)
    out3h,out3v=cfm(out3h,out4v)
    out2h,pred=cfm(out2h,out3v)
    return out2h,out3h,out4h,out5v,pred

def f3net(inputq):
    out2h,out3h,out4h,out5v=ResNet(inputq)
    out2h=basicblock(out2h)
    out3h = basicblock(out3h)
    out4h = basicblock(out4h)
    out5v = basicblock(out5v)
    out2h,out3h,out4h,out5v,pred1=decoder(out2h,out3h,out4h,out5v)
    refine5=resize_like(pred1,out5v)
    refine4 = resize_like(refine5,out4h)
    refine3 = resize_like(refine4,out3h)
    refine2 = resize_like(refine3,out2h)
    out5v=keras.layers.add([out5v,refine5])
    q1=keras.layers.add([out4h,refine4])
    out4h,out4v=cfm(q1,out5v)
    q2=keras.layers.add([out3h, refine3])
    out3h, out3v = cfm(q2, out4v)
    refine2=resize_like(refine2,out2h)
    q=keras.layers.add([out2h, refine2])
    out3v=resize_like(out3v,q)
    q3=keras.layers.add([out2h, refine2])
    out2h, pred2= cfm(q3, out3v)

    pred1= Conv2D(1,kernel_size=3,padding='same',strides=1)(pred1)
    pred2 = Conv2D(1, kernel_size=3, padding='same', strides=1)(pred2)
    out2h = Conv2D(1, kernel_size=3, padding='same', strides=1)(out2h)
    out3h = Conv2D(1, kernel_size=3, padding='same', strides=1)(out3h)
    out4h = Conv2D(1, kernel_size=3, padding='same', strides=1)(out4h)
    out5h = Conv2D(1, kernel_size=3, padding='same', strides=1)(out5v)
    pred1=resize_like(pred1,inputq,name='pred1')
    pred2=resize_like(pred2,inputq,name='pred2')

    out2h=resize_like(out2h,inputq,name='out2h')
    out3h = resize_like(out3h, inputq,name='out3h')
    out4h = resize_like(out4h, inputq,name='out4h')
    out5h = resize_like(out5h, inputq,name='out5h')
    models = Model(input=inputq, output=[pred1, pred2, out2h, out3h, out4h, out5h])
    return models








