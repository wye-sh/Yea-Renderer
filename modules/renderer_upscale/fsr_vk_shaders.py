#!/usr/bin/env python3
"""Compiles AMD's FSR 3.1 GLSL passes (third_party/ffx_vk, FidelityFX SDK v1.1.4) into the permutation headers its
shader-blob accessor includes, as AMD's FidelityFX_SC does on Windows (its CMakeCompileShaders.txt and
CMakeCompileFSR3UpscalerShaders.txt): every pass, every permutation key, in four sets (wave32, wave64, 16-bit,
wave64 16-bit), with each blob's descriptor bindings reflected from its SPIR-V. For GLSL a wave64 set is the
wave32 SPIR-V (the backend asks for 64-wide subgroups when it creates the pipeline) and a 16-bit set adds
FFX_HALF=1. Each distinct SPIR-V is written once, in its own header, which every set that uses it includes.

Run from anywhere, with glslangValidator (Khronos) on PATH: python3 fsr_vk_shaders.py
"""
import concurrent.futures, hashlib, os, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = os.path.join(HERE, 'third_party', 'ffx_vk')
GPU = os.path.join(SDK, 'include', 'FidelityFX', 'gpu')
PASSES = os.path.join(SDK, 'src', 'backends', 'vk', 'shaders', 'fsr3upscaler')
OUT = os.path.join(SDK, 'generated')

BASE = ['--target-env', 'vulkan1.1', '-S', 'comp', '-Os', '-DFFX_GLSL=1', '-DFFX_GPU=1',
        '-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0',
        '-DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0',
        '-DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1',
        '-DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0',
        '-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2', '-I' + GPU, '-I' + os.path.join(GPU, 'fsr3upscaler')]
# The permutation key's bits, lowest first, as the accessor's POPULATE_PERMUTATION_KEY fills them.
OPTIONS = ['FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE', 'FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT',
           'FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS', 'FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS',
           'FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH', 'FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING']
SETS = [('', 0), ('_wave64', 0), ('_16bit', 1), ('_wave64_16bit', 1)]
KINDS = ['CBV', 'TextureSRV', 'TextureUAV', 'BufferSRV', 'BufferUAV', 'Sampler', 'RTAccelerationStructure']


def compile_one(src, half, key):
    defs = ['-D%s=%d' % (o, (key >> i) & 1) for i, o in enumerate(OPTIONS)] + ['-DFFX_HALF=%d' % half]
    with tempfile.TemporaryDirectory() as tmp:
        spv = os.path.join(tmp, 'out.spv')
        r = subprocess.run(['glslangValidator'] + BASE + defs + ['-o', spv, src], capture_output=True, text=True)
        if r.returncode:
            sys.exit('%s key %d half %d:\n%s%s' % (src, key, half, r.stdout, r.stderr))
        return open(spv, 'rb').read()


def reflect(spv):
    """The descriptor bindings, as SPIRV-Reflect sorts them for FidelityFX_SC: (kind, name, binding, count, set)."""
    w = struct.unpack('<%dI' % (len(spv) // 4), spv)
    names, deco, member_nonwritable, types, consts, variables = {}, {}, {}, {}, {}, []
    i = 5
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xFFFF
        a = w[i + 1:i + n]
        if op == 5:  # OpName
            names[a[0]] = bytes(struct.pack('<%dI' % (n - 2), *a[1:])).split(b'\0')[0].decode()
        elif op == 71:  # OpDecorate
            deco.setdefault(a[0], {})[a[1]] = a[2] if len(a) > 2 else True
        elif op == 72 and a[2] == 24:  # OpMemberDecorate NonWritable
            member_nonwritable.setdefault(a[0], set()).add(a[1])
        elif op in (25, 26, 27, 28, 29, 30, 32, 5341):  # image, sampler, sampled image, array, runtime array, struct, pointer, AS
            types[a[0]] = (op, a[1:])
        elif op == 43:  # OpConstant
            consts[a[1]] = a[2]
        elif op == 59:  # OpVariable
            variables.append((a[1], a[0], a[2]))
        i += n
    out = []
    for vid, ptr, storage in variables:
        d = deco.get(vid, {})
        if 33 not in d:  # no Binding: not a descriptor
            continue
        t, count = types[ptr][1][1], 1
        while types[t][0] in (28, 29):
            count = consts[types[t][1][1]] if types[t][0] == 28 else 0
            t = types[t][1][0]
        op, args = types[t]
        td = deco.get(t, {})
        if storage == 0 and op == 25:
            kind = {1: 'TextureSRV', 2: 'TextureUAV'}[args[5]]
        elif storage == 0 and op == 26:
            kind = 'Sampler'
        elif storage == 0 and op == 5341:
            kind = 'RTAccelerationStructure'
        elif storage == 2 and 2 in td:  # Uniform Block
            kind = 'CBV'
        elif storage == 12 or (storage == 2 and 3 in td):  # storage buffer
            members = len(args)
            ro = 24 in d or len(member_nonwritable.get(t, ())) == members
            kind = 'BufferSRV' if ro else 'BufferUAV'
        else:
            sys.exit('unsupported descriptor %s' % names.get(vid))
        out.append((kind, names.get(vid, ''), d[33], count, d.get(34, 0)))
    return sorted(out, key=lambda r: (r[4], r[2]))


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    passes = sorted(f[:-5] for f in os.listdir(PASSES) if f.endswith('.glsl'))
    jobs = {}
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as ex:
        for p in passes:
            for half in (0, 1):
                for key in range(1 << len(OPTIONS)):
                    jobs[(p, half, key)] = ex.submit(compile_one, os.path.join(PASSES, p + '.glsl'), half, key)
        blobs = {k: v.result() for k, v in jobs.items()}
    written = set()
    for p in passes:
        for suffix, half in SETS:
            name = p + suffix
            unique, index = [], {}
            for key in range(1 << len(OPTIONS)):
                h = hashlib.md5(blobs[(p, half, key)]).hexdigest()
                if h not in unique:
                    unique.append(h)
                index[key] = unique.index(h)
                blob_header(p, h, blobs[(p, half, key)], written)
            lines = ['// Generated by fsr_vk_shaders.py from AMD\'s %s.glsl (FidelityFX SDK v1.1.4). Do not edit.' % p, '#pragma once', '#include <stdint.h>']
            lines += ['#include "%s_%s.h"' % (p, h) for h in unique]
            lines += ['', 'typedef union %s_PermutationKey {' % name, '    struct {']
            lines += ['        uint32_t %s : 1;' % o for o in OPTIONS]
            lines += ['    };', '    uint32_t index;', '} %s_PermutationKey;' % name, '']
            lines += ['typedef struct %s_PermutationInfo {' % name, '    const uint32_t       blobSize;', '    const unsigned char* blobData;', '']
            for field in ['constantBuffer', 'srvTexture', 'uavTexture', 'srvBuffer', 'uavBuffer', 'sampler', 'rtAccelerationStructure']:
                num = {'constantBuffer': 'numConstantBuffers', 'srvTexture': 'numSRVTextures', 'uavTexture': 'numUAVTextures',
                       'srvBuffer': 'numSRVBuffers', 'uavBuffer': 'numUAVBuffers', 'sampler': 'numSamplers',
                       'rtAccelerationStructure': 'numRTAccelerationStructures'}[field]
                lines += ['    const uint32_t  %s;' % num, '    const char**    %sNames;' % field, '    const uint32_t* %sBindings;' % field,
                          '    const uint32_t* %sCounts;' % field, '    const uint32_t* %sSpaces;' % field, '']
            lines += ['} %s_PermutationInfo;' % name, '', 'static const uint32_t g_%s_IndirectionTable[] = {' % name]
            lines += ['    %d,' % index[k] for k in range(1 << len(OPTIONS))]
            lines += ['};', '', 'static const %s_PermutationInfo g_%s_PermutationInfo[] = {' % (name, name)]
            for h in unique:
                refl = reflect(blobs_by_hash[h])
                row = '    { g_%s_%s_size, g_%s_%s_data, ' % (p, h, p, h)
                for kind in KINDS:
                    n = sum(1 for r in refl if r[0] == kind)
                    row += ('%d, g_%s_%s_%sResourceNames, g_%s_%s_%sResourceBindings, g_%s_%s_%sResourceCounts, g_%s_%s_%sResourceSets, '
                            % ((n,) + (p, h, kind) * 4)) if n else '0, 0, 0, 0, 0, '
                lines.append(row + '},')
            lines += ['};', '']
            open(os.path.join(OUT, name + '_permutations.h'), 'w').write('\n'.join(lines))
    print('%d passes, %d distinct SPIR-V blobs, %.1f MB of SPIR-V' % (len(passes), len(written), sum(len(blobs_by_hash[h]) for h in written) / 1e6))


blobs_by_hash = {}


def blob_header(p, h, spv, written):
    if h in written:
        return
    written.add(h)
    blobs_by_hash[h] = spv
    lines = ['// Generated by fsr_vk_shaders.py from AMD\'s %s.glsl (FidelityFX SDK v1.1.4). Do not edit.' % p, '#pragma once', '']
    for kind in KINDS:
        rows = [r for r in reflect(spv) if r[0] == kind]
        if rows:
            pre = 'g_%s_%s_%s' % (p, h, kind)
            lines.append('static const char* %sResourceNames[] = { %s };' % (pre, ', '.join('"%s"' % r[1] for r in rows)))
            lines.append('static const uint32_t %sResourceBindings[] = { %s };' % (pre, ', '.join(str(r[2]) for r in rows)))
            lines.append('static const uint32_t %sResourceCounts[] = { %s };' % (pre, ', '.join(str(r[3]) for r in rows)))
            lines.append('static const uint32_t %sResourceSets[] = { %s };' % (pre, ', '.join(str(r[4]) for r in rows)))
    lines += ['', 'static const uint32_t g_%s_%s_size = %d;' % (p, h, len(spv)), '',
              'static const unsigned char g_%s_%s_data[] = {' % (p, h)]
    for i in range(0, len(spv), 32):
        lines.append(','.join(str(b) for b in spv[i:i + 32]) + ',')
    lines += ['};', '']
    open(os.path.join(OUT, '%s_%s.h' % (p, h)), 'w').write('\n'.join(lines))


if __name__ == '__main__':
    main()
