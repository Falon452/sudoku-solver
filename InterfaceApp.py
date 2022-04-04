import kivy
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen

kivy.require('1.9.1')


class StartScreen(Screen):
    def upload(self):
        self.manager.current = 'image'


class ImageScreen(Screen):
    def upload(self):
        self.img1.source = 'sudoku_images/before.jpg'

    def solve(self):
        self.manager.current = 'solved'


class SolvedScreen(Screen):
    def upload(self):
        self.manager.current = 'image'



class InterfaceApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(StartScreen(name='start'))
        sm.add_widget(ImageScreen(name='image'))
        sm.add_widget(SolvedScreen(name='solved'))

        return sm


InterfaceApp().run()