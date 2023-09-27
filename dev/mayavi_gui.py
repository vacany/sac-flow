import numpy as np
from mayavi import mlab

from traits.api import HasTraits, Range, Instance,on_trait_change
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

x, y = np.mgrid[0:3:1,0:3:1]

class MyModel(HasTraits):
    scene = Instance(MlabSceneModel, ())
    slider = Range(0., 10, 0., )
    def __init__(self):
        HasTraits.__init__(self)
        self.all_pc = np.random.rand(10, 10000, 3)



        # self.s = mlab.surf(x, y, np.asarray(x*1.5, 'd'), figure=self.scene.mayavi_scene)
        self.s = mlab.points3d(self.all_pc[0,:,0], self.all_pc[0,:,1], self.all_pc[0,:,2], figure=self.scene.mayavi_scene)

    @on_trait_change('slider')
    def slider_changed(self):
        # self.s.mlab_source.scalars = np.asarray(x * (self.slider + 1), 'd')
        idx = int(self.slider)
        self.s.remove()
        
        self.s = mlab.points3d(self.all_pc[idx, :, 0], self.all_pc[idx, :, 1], self.all_pc[idx, :, 2],
                           figure=self.scene.mayavi_scene)


    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene)),
                Group("slider"))

my_model = MyModel()
my_model.configure_traits()
