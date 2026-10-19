# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""HIP support for the CUDA device branches of Steel's native snippets.

Steel's native snippets guard their device code with ``#if defined(__CUDA_ARCH__)``
and use the CUDA warp-synchronous intrinsics with 32-bit lane masks, plus a little
inline PTX (``prefetch.global.L1`` and the ``cp.async`` shared-memory staging). HIP's
``*_sync`` intrinsics require 64-bit masks and HIP has no PTX, so the snippets would
not compile there as written. :func:`func_native` lets the device branch compile for
HIP too: it extends each guard to HIP device compilation, maps the 32-bit-mask
intrinsics onto HIP's mask-free ones, and routes the inline PTX through macros that
expand to the original statement on CUDA and to a plain equivalent on HIP (a prefetch
builtin; synchronous LDS copies in place of ``cp.async``, whose commit and wait become
no-ops). The mapping assumes 32-lane wavefronts, as the snippets do.

On CUDA and on the CPU the preprocessed snippet is unchanged.
"""

import re

import warp as wp

_CUDA_GUARD = "#if defined(__CUDA_ARCH__)"
_DEVICE_GUARD = "#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)"

# Inline PTX statements as the snippets write them, and the macro each becomes.
_PTX_REWRITES = (
    (
        re.compile(r'asm volatile\("prefetch\.global\.L1 \[%0\];" :: "l"\(([^;]+?)\)\);'),
        r"DEVICE_PREFETCH_L1(\1);",
    ),
    (
        re.compile(
            r'asm volatile\("cp\.async\.ca\.shared\.global \[%0\], \[%1\], (4|16);\\n" :: '
            r'"r"\((\w+)\), "l"\((\w+)\) : "memory"\);'
        ),
        r"DEVICE_CP_ASYNC_\1(\2, \3);",
    ),
    (re.compile(r'asm volatile\("cp\.async\.commit_group;\\n" ::: "memory"\);'), "DEVICE_CP_ASYNC_COMMIT();"),
    (re.compile(r'asm volatile\("cp\.async\.wait_all;\\n" ::: "memory"\);'), "DEVICE_CP_ASYNC_WAIT_ALL();"),
)

_PROLOGUE = r"""
#if defined(__HIP_DEVICE_COMPILE__)
#define __syncwarp(...) __syncwarp()
#define __ballot_sync(mask, predicate) ((unsigned int)(__ballot(predicate) & (mask)))
#define __any_sync(mask, predicate) (((__ballot(predicate) & (mask)) != 0) ? 1 : 0)
#define __all_sync(mask, predicate) (((__ballot(predicate) & (mask)) == (mask)) ? 1 : 0)
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
// A generic pointer into LDS carries the LDS byte offset in its low 32 bits; the staging
// code keeps such offsets in an unsigned and hands them back for the copies.
typedef int device_i4_t __attribute__((ext_vector_type(4)));
#define __cvta_generic_to_shared(p) ((size_t)(unsigned)(size_t)((__attribute__((address_space(3))) const char*)(p)))
#define DEVICE_LDS_PTR(T, off) ((__attribute__((address_space(3))) T*)(size_t)(unsigned)(off))
#define DEVICE_CP_ASYNC_4(dst, src) (*DEVICE_LDS_PTR(int, dst) = *static_cast<const int*>(src))
#define DEVICE_CP_ASYNC_16(dst, src) (*DEVICE_LDS_PTR(device_i4_t, dst) = *static_cast<const device_i4_t*>(src))
#define DEVICE_CP_ASYNC_COMMIT() ((void)0)
#define DEVICE_CP_ASYNC_WAIT_ALL() ((void)0)
#define DEVICE_PREFETCH_L1(p) __builtin_prefetch((p))
#else
#define DEVICE_CP_ASYNC_4(dst, src) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src) : "memory")
#define DEVICE_CP_ASYNC_16(dst, src) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory")
#define DEVICE_CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n" ::: "memory")
#define DEVICE_CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::: "memory")
#define DEVICE_PREFETCH_L1(p) asm volatile("prefetch.global.L1 [%0];" :: "l"(p))
#endif
"""

_EPILOGUE = r"""
#if defined(__HIP_DEVICE_COMPILE__)
#undef __syncwarp
#undef __ballot_sync
#undef __any_sync
#undef __all_sync
#undef __shfl_sync
#undef __shfl_up_sync
#undef __shfl_down_sync
#undef __shfl_xor_sync
#undef __cvta_generic_to_shared
#undef DEVICE_LDS_PTR
#endif
#undef DEVICE_CP_ASYNC_4
#undef DEVICE_CP_ASYNC_16
#undef DEVICE_CP_ASYNC_COMMIT
#undef DEVICE_CP_ASYNC_WAIT_ALL
#undef DEVICE_PREFETCH_L1
"""


def hip_snippet(snippet: str) -> str:
    """Return ``snippet`` with its CUDA device branches enabled for HIP."""
    if _CUDA_GUARD not in snippet:
        return snippet
    text = snippet.replace(_CUDA_GUARD, _DEVICE_GUARD)
    for pattern, replacement in _PTX_REWRITES:
        text = pattern.sub(replacement, text)
    return _PROLOGUE + text + _EPILOGUE


def func_native(snippet: str, adj_snippet: str | None = None, replay_snippet: str | None = None):
    """:func:`warp.func_native` for snippets whose device code is CUDA-only."""
    return wp.func_native(
        hip_snippet(snippet),
        None if adj_snippet is None else hip_snippet(adj_snippet),
        None if replay_snippet is None else hip_snippet(replay_snippet),
    )
