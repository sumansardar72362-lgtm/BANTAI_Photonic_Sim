import sys
import os
import numpy as np
import pytest

# মেইন ফোল্ডারের পাথ যুক্ত করা হচ্ছে
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.kecia.core.glass_memory import GlassMemoryMapper
from src.kecia.physics.optical_mac import OpticalMAC
from src.kecia.core.tensor5d import Tensor5D
from src.kecia.nn.linear5d import Linear5D
from src.kecia.nn.activation import PhotonicReLU
def test_glass_memory_conversion():
    """টেস্ট ১: 5D মেমোরি ঠিকমতো আলোতে কনভার্ট ও রিড করতে পারে কি না"""
    mapper = GlassMemoryMapper(grid_x=3, grid_y=3, layers_z=1)
    input_data = np.array([[1.5, -2.0], [-3.1, 4.0]])
    
    mapper.encode_to_light(input_data, z_layer=0)
    output_data = mapper.read_from_light(2, 2, z_layer=0)
    
    assert np.allclose(input_data, output_data), "Glass Memory Conversion Failed!"

def test_optical_mac_math():
    """টেস্ট ২: আলোর ফিজিক্স ব্যবহার করে গুণ ঠিকমতো হচ্ছে কি না"""
    inputs = np.array([1.5, -2.0, 3.0])
    weights = np.array([[2.0, -1.0], [-3.0, 4.0], [1.5, 0.0]])
    
    in_int = np.abs(inputs)
    in_ph = np.where(inputs < 0, np.pi, 0.0)
    w_int = np.abs(weights)
    w_ph = np.where(weights < 0, np.pi, 0.0)
    
    mac = OpticalMAC()
    optical_out = mac.compute_vmm(in_int, in_ph, w_int, w_ph)
    cpu_out = np.dot(inputs, weights)
    
    assert np.allclose(optical_out, cpu_out), "Optical MAC Math is incorrect!"

def test_tensor5d_fallback():
    """টেস্ট ৩: Tensor5D 3-tier ফলব্যাক ঠিকমতো কাজ করে কি না"""
    a = Tensor5D([1.0, 2.0], device='photonic')
    b = Tensor5D([[2.0, 1.0], [-1.0, 3.0]], device='photonic')
    
    res_photonic = a @ b
    res_cpu = a.to('cpu') @ b.to('cpu')
    
    assert res_photonic.device == 'photonic'
    assert res_cpu.device == 'cpu'
    assert np.allclose(res_photonic.data, res_cpu.data), "Tensor Fallback Logic Failed!"

def test_linear5d_layer():
    """টেস্ট ৪: নিউরাল নেটওয়ার্ক লেয়ার ঠিকমতো কাজ করে কি না"""
    layer = Linear5D(in_features=3, out_features=2, device='photonic')
    inputs = Tensor5D([1.0, 2.0, -1.0])
    
    out_photonic = layer(inputs)
    out_cpu = layer.to('cpu')(inputs.to('cpu'))
    
    assert np.allclose(out_photonic.data, out_cpu.data, atol=1e-3), "Linear Layer outputs mismatch between Photonic and CPU!"


def test_photonic_relu():
    """টেস্ট ৫: ফোটোনিক অপটিক্যাল থ্রেশোল্ডিং (ReLU) ঠিকমতো কাজ করে কি না"""
    relu = PhotonicReLU()
    inputs = Tensor5D([-2.5, 0.0, 3.14, -1.1, 5.0], device='photonic')
    
    out_photonic = relu(inputs)
    out_cpu = relu(inputs.to('cpu'))
    
    # নেগেটিভ ভ্যালুগুলো ০ হয়ে যাবে, পজিটিভগুলো ঠিক থাকবে
    assert np.allclose(out_photonic.data, out_cpu.data, atol=1e-3), "Photonic ReLU mismatch!"