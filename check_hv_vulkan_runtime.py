#!/usr/bin/env python3
"""Check graphics driver loading and Vulkan GPU enumeration without CARLA."""
import ctypes as C
import json
import os
import sys


class InstanceCreateInfo(C.Structure):
    _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('flags', C.c_uint32),
                ('pApplicationInfo', C.c_void_p), ('enabledLayerCount', C.c_uint32),
                ('ppEnabledLayerNames', C.c_void_p), ('enabledExtensionCount', C.c_uint32),
                ('ppEnabledExtensionNames', C.c_void_p)]


def enumerate_devices(vulkan):
    create = vulkan.vkCreateInstance
    create.argtypes = [C.POINTER(InstanceCreateInfo), C.c_void_p, C.POINTER(C.c_void_p)]
    create.restype = C.c_int32
    destroy = vulkan.vkDestroyInstance
    destroy.argtypes = [C.c_void_p, C.c_void_p]
    destroy.restype = None
    enumerate_gpu = vulkan.vkEnumeratePhysicalDevices
    enumerate_gpu.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)]
    enumerate_gpu.restype = C.c_int32
    properties = vulkan.vkGetPhysicalDeviceProperties
    properties.argtypes = [C.c_void_p, C.c_void_p]
    properties.restype = None
    instance = C.c_void_p()
    info = InstanceCreateInfo(sType=1)
    result = create(C.byref(info), None, C.byref(instance))
    report = {'create_instance_result':result, 'devices':[]}
    if result != 0:
        return report
    try:
        count = C.c_uint32()
        result = enumerate_gpu(instance, C.byref(count), None)
        report.update(enumerate_count_result=result, device_count=count.value)
        if result != 0 or not count.value:
            return report
        if count.value > 256:
            raise RuntimeError('Unexpected physical device count: %d'%count.value)
        devices = (C.c_void_p*count.value)()
        result = enumerate_gpu(instance, C.byref(count), devices)
        report['enumerate_devices_result'] = result
        if result != 0:
            return report
        for device in devices[:count.value]:
            # Vulkan 1.0 VkPhysicalDeviceProperties has a fixed ABI. Reserve
            # space for the entire structure; read only its fixed header.
            storage = (C.c_uint64*512)()
            properties(device, C.byref(storage))
            raw = C.string_at(C.addressof(storage), 276)
            vendor = int.from_bytes(raw[8:12], sys.byteorder)
            name = raw[20:276].split(b'\0',1)[0].decode('utf-8',errors='replace')
            report['devices'].append({'vendor_id':hex(vendor), 'name':name,
                                      'nvidia':vendor == 0x10de})
        return report
    finally:
        destroy(instance, None)


def probe(load_library=None):
    load_library = load_library or C.CDLL
    report = {'uid':os.getuid() if hasattr(os,'getuid') else None,
              'NVIDIA_DRIVER_CAPABILITIES':os.environ.get('NVIDIA_DRIVER_CAPABILITIES'),
              'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH'),
              'libraries':{}, 'passed':False,
              'scope':'Driver loading and GPU enumeration only; not a CARLA startup certificate.'}
    handles = {}
    for name in ('libGLX_nvidia.so.0', 'libvulkan.so.1'):
        try:
            handles[name] = load_library(name)
            report['libraries'][name] = {'loaded':True}
        except OSError as exc:
            report['libraries'][name] = {'loaded':False, 'error':str(exc)}
    if len(handles) != 2:
        report['failure'] = 'Required graphics driver/loader library cannot be loaded.'
        return report
    try:
        report['vulkan'] = enumerate_devices(handles['libvulkan.so.1'])
        report['passed'] = any(d['nvidia'] for d in report['vulkan']['devices'])
        if not report['passed']:
            report['failure'] = 'Vulkan did not enumerate an NVIDIA physical device.'
    except Exception as exc:
        report['failure'] = '%s: %s'%(type(exc).__name__,exc)
    return report


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run this check as the CARLA server user on Linux.')
    report = probe()
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
