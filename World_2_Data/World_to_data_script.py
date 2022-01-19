# -*- coding: utf-8 -*-
__version__ = '0.1'

'''
Created on Fri Oct 29 09:32:34 2021

@author: Marc Boucsein
'''

import numpy as np
import dask.array as da
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
import napari.types
from napari.types import LayerDataTuple
from typing import TYPE_CHECKING
import gryds
if TYPE_CHECKING:
    import napari
    

from concurrent.futures import Future
from skimage import img_as_float32, img_as_ubyte, img_as_uint, img_as_int


def on_init(widget):
    '''
    Initializes widget layout.
    Updates widget layout according to user input.
    '''
    widget.native.setStyleSheet('QWidget{font-size: 12pt;}')
    
    setattr(getattr(widget, 'Line_Edit'), 'visible', False)
    
    
    def update_widget(event):
        if widget.layer_choice.value=='All Layers [prefix]':
            setattr(getattr(widget, 'Line_Edit'), 'label', 'Prefix:')
            setattr(getattr(widget, 'Line_Edit'), 'visible', True)
        elif widget.layer_choice.value=='All Layers [suffix]':
            setattr(getattr(widget, 'Line_Edit'), 'label', 'Suffix:')
            setattr(getattr(widget, 'Line_Edit'), 'visible', True)
        else:
            setattr(getattr(widget, 'Line_Edit'), 'visible', False)


    widget.layer_choice.changed.connect(update_widget)
    widget.native.layout().addStretch()



@magic_factory(widget_init=on_init, layout='horizontal', call_button='Convert',
               Order={'max': 5,'tooltip': 'Select an interpolation order'},
               layer_choice={'widget_type': 'RadioButtons', 'label': 'Choices:', 'choices': ['All Layers','All Layers [prefix]', 'All Layers [suffix]', 'Selected'], 'value': 'Selected',
                'tooltip': 'Select all layers, all layers with given prefix or suffix or select just layers from the layer list'}, 
               Line_Edit={'widget_type': 'LineEdit', 'label': 'Prefix:'}, 
               Checkbox_trivial={'widget_type': 'Checkbox', 'value':False, 'label': 'Trivial', 'tooltip': 'This option ignores all transformation information and makes the transformation trivial.'},
               Checkbox_datatype={'widget_type': 'Checkbox', 'value':True, 'label': 'Keep datatype', 'tooltip': 'This option maintains the original datatype after applying the transformation'},
             )

def World2Data(viewer: 'napari.viewer.Viewer', Order: int, layer_choice: str, Line_Edit: str, Checkbox_trivial: bool, Checkbox_datatype: bool) -> Future[LayerDataTuple]:
    from napari.qt import thread_worker
    future: Future[LayerDataTuple] = Future()
    
    def _on_done(result, self=World2Data):
        future.set_result(result)
    
    
    @thread_worker
    def worldtodata():
        if layer_choice=='All Layers [prefix]':
            list_1=[x  for x in viewer.layers if (isinstance(x, napari.layers.image.image.Image)
                                                  and x.name.startswith(str(Line_Edit)))]
            
            if not list_1:
                print('No layers found with given prefix')
                return
            
        elif layer_choice=='All Layers [suffix]':
            list_1=[x  for x in viewer.layers if (isinstance(x, napari.layers.image.image.Image)
                                                  and x.name.endswith(str(Line_Edit)))]
            
            if not list_1:
                print('No layers found with given suffix') 
                return
            
        elif layer_choice=='All Layers':
            list_1=[x  for x in viewer.layers if (isinstance(x, napari.layers.image.image.Image))]
            
            if not list_1:
                print('No layers found')
                return
            
        else:
            list_1=[s for s in viewer.layers.selection if isinstance(s, napari.layers.image.image.Image)]
            
            if not list_1:
                print('Select at least one layer')
                return 
        
        layer_list=[] 
        if Checkbox_trivial: 
            for layer in list_1: 
                img_data=layer.data
                if isinstance(img_data, da.core.Array):
                    img_data=img_data.compute()
                img_shape=img_data.shape 
                img_rgb=layer.rgb
                img_name=layer.name
                Layer_tuple= (img_data, {'name': img_name+'_data', 'rgb': img_rgb})
                layer_list.append(Layer_tuple)
            return layer_list
        
        if Checkbox_datatype:
            datatypes = [l.data.dtype for l in list_1]
        
        for i, layer in enumerate(list_1):
            img_data=layer.data
            if isinstance(img_data, da.core.Array):
                img_data=img_data.compute()
            img_data=img_as_float32(img_data)
            img_shape=img_data.shape
            img_name=layer.name
            img_affine=layer.affine.affine_matrix
            img_rotate=layer.rotate
            img_scale=layer.scale
            img_shear=layer.shear
            img_translate=layer.translate
            img_rgb=layer.rgb
            
            affine_matrix_ind=napari.utils.transforms.transforms.Affine(rotate= img_rotate, scale=img_scale, shear=img_shear, translate=img_translate).affine_matrix
            if len(img_shape)==2:
                img_affine[0][2]=img_affine[0][2] / img_shape[0]
                img_affine[1][2]=img_affine[1][2] / img_shape[1]
                affine_matrix_ind[0][2]=affine_matrix_ind[0][2] / img_shape[0]
                affine_matrix_ind[1][2]=affine_matrix_ind[1][2] / img_shape[1]
            elif len(img_shape)==3 and img_rgb:
                img_affine[0][2]=img_affine[0][2] / img_shape[0]
                img_affine[1][2]=img_affine[1][2] / img_shape[1]
                affine_matrix_ind[0][2]=affine_matrix_ind[0][2] / img_shape[0]
                affine_matrix_ind[1][2]=affine_matrix_ind[1][2] / img_shape[1]
            elif len(img_shape)==3 and not img_rgb:
                img_affine[0][3]=img_affine[0][3] / img_shape[0]
                img_affine[1][3]=img_affine[1][3] / img_shape[1]
                img_affine[2][3]=img_affine[2][3] / img_shape[2]
                affine_matrix_ind[0][3]=affine_matrix_ind[0][3] / img_shape[0]
                affine_matrix_ind[1][3]=affine_matrix_ind[1][3] / img_shape[1]
                affine_matrix_ind[2][3]=affine_matrix_ind[2][3] / img_shape[1]
            else:
                print('Just 2D/3D support: One of the provided layer are not 2D or 3D')
                return
            
            aff_mat=img_affine @ affine_matrix_ind
            aff_mat_inv=np.linalg.inv(aff_mat)
            aff_mat_del= aff_mat_inv[:-1,:] 
            if len(img_shape)==2:
                affine_2D = gryds.AffineTransformation(ndim=2, affine=aff_mat_del)
                interpolator = gryds.Interpolator(img_data)
                transformed_image = interpolator.transform(affine_2D, order=Order)
            elif len(img_shape)==3 and img_rgb:
                affine_2D = gryds.AffineTransformation(ndim=2, affine=aff_mat_del)
                transformed_image=np.zeros_like(img_data)
                for i in range(3): 
                    interpolator = gryds.Interpolator(img_data[:,:,i])
                    transformed_image_rgb = interpolator.transform(affine_2D, order=Order)
                    transformed_image[:,:,i]=transformed_image_rgb
            elif len(img_shape)==3 and not img_rgb:
                affine_3D = gryds.AffineTransformation(ndim=3,  affine=aff_mat_del)
                interpolator = gryds.Interpolator(img_data)
                transformed_image = interpolator.transform(affine_3D, order=Order) 
            
            
            if Checkbox_datatype:
                dtype = datatypes[i]
                if dtype == 'uint8':
                    dtype_transform = img_as_ubyte
                elif dtype == 'uint16':
                    dtype_transform = img_as_uint
                elif dtype == 'int16':
                    dtype_transform = img_as_int
                elif dtype == 'float32':
                    dtype_transform = img_as_float32
                else: 
                    print(f"Unknown datatype: {dtype}, we return a float32 image")
                    dtype_transform = img_as_float32
                    
                transformed_image = dtype_transform(transformed_image)
            
            #add the transformed image to layers
            Layer_tuple= (transformed_image, {'name': img_name+'_data', 'rgb': img_rgb})
            layer_list.append(Layer_tuple)
        
        
        return layer_list
    
    
    worker = worldtodata()
    worker.returned.connect(_on_done)
    worker.start()
    
    return future


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return World2Data


