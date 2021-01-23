"""
MIT License

Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIWND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow
from tensorflow.keras import layers, Model

from .common import YOLOConv2D

class BiFPNTiny(Model):
    def __init__(
        self, num_classes, activation: str = "mish", kernel_regularizer=None
    ):
        super(BiFPNTiny, self).__init__(name="BiFPNTiny")
        self.conv18 = YOLOConv2D(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.upSampling18 = layers.UpSampling2D(interpolation="bilinear")

        self.concat13_18 = layers.Concatenate(axis=-1)

        self.conv19 = YOLOConv2D(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        self.concat13_19 = layers.Concatenate(axis=-1)
        self.conv20 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )
        self.maxpool20 = layers.MaxPool2D((2, 2), strides=2, padding="same")
        self.concat13_20 = layers.Concatenate(axis=-1)
        self.conv21 = YOLOConv2D(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
        )   

    def call(self, x):
        route1, route2 = x #(None,32,32,256),(None,16,16,512)

        x1 = self.conv18(route2) #(None,16, 16, 512)

        #pred_l = self.conv19(x1) #(None, 16, 16, 48)

        x2 = self.upSampling18(route2) #(None,32, 32, 512)
        
        x2 = self.concat13_18([x2, route1]) #(None, 32, 32, 768)
        
        x2 = self.conv19(x2) #(None, 32, 32, 256)

        x2 = self.concat13_19([x2, route1]) #(None, 32, 32, 512)
        
        x3 = self.conv20(x2)
        pred_m = x3 #(None, 32, 32, 48)

        x2 = self.maxpool20(x2)

        x2 = self.concat13_20([x2, x1])

        x2 = self.conv21(x2) #(None, 16, 16, 48)

        pred_l = x2
        return pred_m, pred_l #(None, 32, 32, 48), (None, 16, 16, 48)