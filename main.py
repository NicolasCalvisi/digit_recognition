from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Line
from kivy import platform
from kivy.core.window import Window
from kivy.uix.popup import Popup
from time import sleep
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from  kivy.uix.label import Label
from PIL import Image
from os import remove,getcwd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


def file_to_minst(filename):
    img=plt.imread( filename)
    width, height, _ =img.shape
    x_origin=(0.2989*img[:int(0.7*width),:,0]+0.5870*img[:int(0.7*width),:,1]+0.1140 *img[:int(0.7*width),:,2])*img[:int(0.7*width),:,3]
    x_origin.astype(np.float32)
    x=np.zeros([28,28],dtype=np.float32)
    for i in range(28):
        for j in range(28):
            x[i,j]=np.sum(x_origin[int(i*width/28):int((i+1)*width/28),int(j*height/28):int((j+1)*height/28)])

    xnew=np.empty_like(x)
    for i in range(28):xnew[i]=x[(i-4)%28]
    x=xnew.copy()
    x.resize((1,28,28,1))
    x=np.sqrt(x)
    trunc= max(2,np.quantile(x, 0.97))
    x[x>trunc]=trunc
    x/=np.max(x)
    x.astype(np.float32)
    #plt.imshow(x[0,:,:,0])
    #plt.show()
    return x

np.set_printoptions(precision=3,suppress=True)

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:
            Color((1, 1, 1), mode='hsv')
            touch.ud['line'] = Line(points=(touch.x, touch.y),width=6)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MyPaintApp(App):

    def photo(self,obj):

        s=''
        nb=len(str(self.n))
        for i in range(4-nb):s+='0'


        if platform == 'android'or platform == 'linux':
            Window.screenshot(name='./img.png')
            numpy_image=file_to_minst('./img'+s+str(self.n)+'.png' )
            remove('./img'+s+str(self.n)+'.png' )

        if platform == 'win':
            Window.screenshot(name='img.png')
            numpy_image=file_to_minst('img'+s+str(self.n)+'.png' )
            remove('img'+s+str(self.n)+'.png')

        res=(self.model.predict(numpy_image))
        self.prediction="The result is "+str(np.argmax(res[0]))
        self.label.text=self.prediction


    def build(self):
        print(tf.__version__)
        self.model=load_model('my_model.h5')
        self.n=1
        self.prediction=''
        self.superBox  = BoxLayout(orientation='vertical')
        self.Box1      = BoxLayout(orientation='vertical', size_hint=(1,.8))
        self.Box2      = BoxLayout(orientation='horizontal', size_hint=(1,.2) )

        self.label         = Label(text=self.prediction)
        self.Box2.add_widget(self.label)

        self.parent = Widget()
        self.painter = MyPaintWidget()

        self.clearbtn = Button(text='Clear')
        self.clearbtn.bind(on_release=self.clear_canvas)
        self.Box2.add_widget(self.clearbtn)

        self.predictbtn = Button(text='Predict')
        self.predictbtn.bind(on_release=self.photo)
        self.Box2.add_widget(self.predictbtn)
        self.Box1.add_widget(self.painter)

        self.superBox.add_widget(self.Box1)
        self.superBox.add_widget(self.Box2)

        return self.superBox

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.prediction=""
        self.label.text=self.prediction



if __name__ == '__main__':
    MyPaintApp().run()
