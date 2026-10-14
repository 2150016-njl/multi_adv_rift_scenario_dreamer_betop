import ctypes as C
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

from check_hv_vulkan_runtime import enumerate_devices,probe


def fake_vulkan(vendor=0x10de,create_result=0):
    def create(info,allocator,instance):
        C.cast(instance,C.POINTER(C.c_void_p))[0]=C.c_void_p(1)
        return create_result
    def enumerate_gpu(instance,count,devices):
        C.cast(count,C.POINTER(C.c_uint32))[0]=1
        if devices is not None:devices[0]=2
        return 0
    def properties(device,storage):
        raw=bytearray(276)
        raw[8:12]=vendor.to_bytes(4,'little')
        raw[20:24]=b'Test'
        C.memmove(storage,bytes(raw),len(raw))
    return SimpleNamespace(vkCreateInstance=MagicMock(side_effect=create),
        vkDestroyInstance=MagicMock(),vkEnumeratePhysicalDevices=MagicMock(side_effect=enumerate_gpu),
        vkGetPhysicalDeviceProperties=MagicMock(side_effect=properties))


class VulkanRuntime(unittest.TestCase):
    def test_missing_nvidia_library_is_reported_without_vulkan_call(self):
        def load(name):
            if name=='libGLX_nvidia.so.0':raise OSError('cannot open shared object file')
            return object()
        with patch('check_hv_vulkan_runtime.enumerate_devices') as enumerate_gpu:
            result=probe(load)
        self.assertFalse(result['passed'])
        self.assertFalse(result['libraries']['libGLX_nvidia.so.0']['loaded'])
        enumerate_gpu.assert_not_called()

    def test_successful_probe_identifies_nvidia_and_destroys_instance(self):
        api=fake_vulkan()
        result=probe(lambda name:api)
        self.assertTrue(result['passed'])
        self.assertEqual(result['vulkan']['devices'][0]['name'],'Test')
        api.vkDestroyInstance.assert_called_once()

    def test_software_or_other_vendor_device_is_not_nvidia_success(self):
        api=fake_vulkan(vendor=0x10005)
        self.assertFalse(probe(lambda name:api)['passed'])
        api.vkDestroyInstance.assert_called_once()

    def test_create_failure_does_not_destroy_nonexistent_instance(self):
        api=fake_vulkan(create_result=-9)
        result=enumerate_devices(api)
        self.assertEqual(result['create_instance_result'],-9)
        api.vkDestroyInstance.assert_not_called()
        api.vkEnumeratePhysicalDevices.assert_not_called()

    def test_enumeration_error_still_destroys_instance(self):
        api=fake_vulkan()
        api.vkEnumeratePhysicalDevices.side_effect=RuntimeError('driver failed')
        result=probe(lambda name:api)
        self.assertFalse(result['passed'])
        self.assertIn('driver failed',result['failure'])
        api.vkDestroyInstance.assert_called_once()


if __name__=='__main__':unittest.main()
