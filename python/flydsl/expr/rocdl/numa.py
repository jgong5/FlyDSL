# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Die-affine placement and scheduling on two-die GPUs.

gfx1250 in SPX mode interleaves memory between its two dies in chunks: the die
of a byte is ``parity(addr & color_mask)`` (``0x04924000``: 16 KiB chunks). A
die-split tensor stores the first ``split`` elements of its logical layout on
die 0 and the rest on die 1, and a kernel whose die-d workgroups take only part
d's tiles reads local memory. Three pieces, all driven by a ``Topology``:

- placement: ``die_split_tensor`` views a buffer from ``pack_die_split`` (or
  ``empty``) through a composed layout, so every ``fx.copy`` applies it whatever
  the tile shape;
- scheduling: ``die_tile`` gives each workgroup a tile of its own die's part;
- queries: ``xcc_id`` / ``die_id`` on the device, ``empty`` for memory whose
  2 MiB page phase matches its virtual address (the parity rule holds on
  physical addresses).

``topology()`` is the real gfx1250 one; its device queries and ``empty`` are
stubs until the gfx1250 interfaces are named. Other targets run only with
``topology(emulate="gfx1250")``: the gfx1250 rule applied arithmetically, the
die taken from ``HW_REG_XCC_ID`` (gfx942/gfx95x), and an aligned base instead of
aligned page phases. Emulation checks correctness, not locality.

Usage::

    from flydsl.expr.rocdl import numa

    topo = numa.topology(emulate="gfx1250")                 # host
    wp = numa.pack_die_split(w, split, topo)                 # host, once
    W = numa.die_split_tensor(Wp, layout, split, topo)        # @flyc.jit wrapper
    tile = numa.die_tile(tiles, topo)                         # @flyc.kernel
"""

import dataclasses

from ..._mlir.dialects import llvm as _llvm
from ...runtime.device import get_rocm_arch
from .. import gpu
from ..meta import dsl_loc_tracing
from ..numeric import Int32
from ..primitive import get_iter, make_composed_layout, make_layout, make_view, static
from ..typing import SwizzleType, T, Tensor

__all__ = [
    "Topology",
    "die_id",
    "die_of_address",
    "die_split_layout",
    "die_split_offsets",
    "die_split_tensor",
    "die_tile",
    "empty",
    "pack_die_split",
    "topology",
    "xcc_id",
]

_EMULATION_HOSTS = ("gfx942", "gfx95")  # HW_REG_XCC_ID is known there


@dataclasses.dataclass(frozen=True)
class Topology:
    """A two-die GPU: its XCCs (numbered die by die) and its memory interleave."""

    arch: str
    num_xccs: int
    color_mask: int = 0x04924000
    emulated: bool = False

    @property
    def xccs_per_die(self):
        return max(self.num_xccs // 2, 1)

    @property
    def chunk_bytes(self):
        return self.color_mask & -self.color_mask

    @property
    def base_align(self):
        """Offset bits up to the mask's highest must equal the address bits."""
        return 1 << self.color_mask.bit_length()


def _num_xccs(device):
    import ctypes

    from ...runtime.device_runtime import rocm as _rt

    dev = _rt._hip_get_device() if device is None else device
    n = ctypes.c_int(0)
    if _rt._HIP_LIB is None or _rt._HIP_LIB.hipDeviceGetAttribute(ctypes.byref(n), 10018, dev) != 0:
        raise RuntimeError("numa: cannot query the XCC count through HIP (hipDeviceAttributeNumberOfXccs)")
    return n.value


def topology(device=None, *, emulate=None):
    """The device's die topology.

    On gfx1250 this is the real topology. Elsewhere it raises unless
    ``emulate="gfx1250"`` asks for the emulated one (gfx942/gfx95x only).
    """
    arch = get_rocm_arch()
    if arch.startswith("gfx1250"):
        return Topology(arch, _num_xccs(device))
    if emulate != "gfx1250":
        raise NotImplementedError(f"numa: no die topology on {arch!r}; pass emulate='gfx1250' to emulate one")
    if not arch.startswith(_EMULATION_HOSTS):
        raise NotImplementedError(f"numa: cannot emulate gfx1250 on {arch!r} (needs HW_REG_XCC_ID: gfx942/gfx95x)")
    return Topology(arch, _num_xccs(device), emulated=True)


def _stub(what):
    raise NotImplementedError(f"numa: the gfx1250 {what} interface is not available yet")


# ---------------------------------------------------------------- host side


def _chunk_shift(topo, elem_bytes):
    if elem_bytes <= 0 or topo.chunk_bytes % elem_bytes:
        raise ValueError(f"element size {elem_bytes} does not divide a {topo.chunk_bytes}-byte chunk")
    return (topo.chunk_bytes // elem_bytes).bit_length() - 1


def _check_split(split, elem_bytes, topo):
    c = _chunk_shift(topo, elem_bytes)
    if split <= 0 or split % (1 << c):
        raise ValueError(f"split {split} must be a positive multiple of {1 << c} elements (one chunk)")
    return c


def _fold_shifts(topo):
    """Chunk-index bits that the color folds into bit 0 (the die bit)."""
    rel = topo.color_mask // topo.chunk_bytes
    return [b for b in range(1, rel.bit_length()) if rel >> b & 1]


def die_split_offsets(offsets, split, elem_bytes, topo):
    """Physical element offsets of logical ones (Python ints or integer tensors)."""
    c = _check_split(split, elem_bytes, topo)
    d = (offsets >= split) * 1
    off = offsets - d * split
    x = ((off >> c) << 1) | d
    fold = 0
    for s in _fold_shifts(topo):
        fold = fold ^ (x >> s)
    x = x ^ (fold & 1)
    return (x << c) | (off & ((1 << c) - 1))


def die_of_address(addr, topo):
    """Die owning a byte address under the topology's interleave (ints or integer tensors)."""
    m = addr & topo.color_mask
    p = 0
    for b in range(topo.color_mask.bit_length()):
        if topo.color_mask >> b & 1:
            p = p ^ (m >> b)
    return p & 1


def empty(numel, dtype, device, topo):
    """A 1-D torch buffer whose die placement follows ``die_of_address`` on its addresses.

    gfx1250 needs pages whose phase matches the virtual address, from an interface
    that is not available yet. Emulated, the base is aligned to ``topo.base_align``.
    """
    import torch

    if not topo.emulated:
        _stub("phase-aligned allocation")
    eb = torch.empty((), dtype=dtype).element_size()
    raw = torch.empty(numel * eb + topo.base_align, dtype=torch.uint8, device=device)
    skip = -raw.data_ptr() % topo.base_align
    return raw[skip : skip + numel * eb].view(dtype)


def pack_die_split(w, split, topo):
    """Copy a torch tensor into a die-split buffer: the first ``split`` elements of
    ``w`` (flattened) on die 0, the rest on die 1. View it with ``die_split_tensor``."""
    import torch

    eb = w.element_size()
    _check_split(split, eb, topo)
    if split >= w.numel():
        raise ValueError(f"split {split} must be smaller than the tensor ({w.numel()} elements)")
    n_phys = 2 * max(split, w.numel() - split)
    n_phys += -n_phys % (2 * topo.chunk_bytes // eb)
    out = empty(n_phys, w.dtype, w.device, topo)
    out[die_split_offsets(torch.arange(w.numel(), device=w.device), split, eb, topo)] = w.reshape(-1)
    return out


# ---------------------------------------------------------------- device side


@dsl_loc_tracing
def die_split_layout(layout, split, elem_bytes, topo):
    """Compose ``layout`` (logical coordinates -> element offsets) with the die split."""
    c = _check_split(split, elem_bytes, topo)
    ch = 1 << c
    lay = make_composed_layout(make_layout((ch, split // ch, 2), (1, 2 * ch, ch)), layout)
    for s in _fold_shifts(topo):
        lay = make_composed_layout(static(SwizzleType.get(1, c, s)), lay)
    return lay


@dsl_loc_tracing
def die_split_tensor(packed, layout, split, topo):
    """View a die-split buffer (from ``pack_die_split``) with logical ``layout``."""
    elem_bytes = packed.element_type.width // 8
    return Tensor(make_view(get_iter(packed), die_split_layout(layout, split, elem_bytes, topo)))


def _hw_xcc_id():
    xcc = _llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)", "=s", has_side_effects=True)
    return Int32(xcc)


@dsl_loc_tracing
def xcc_id(topo):
    """This wave's XCC, numbered die by die."""
    if not topo.emulated:
        _stub("XCC id")
    return _hw_xcc_id()


@dsl_loc_tracing
def die_id(topo):
    """This wave's die."""
    if not topo.emulated:
        _stub("die id")
    return _hw_xcc_id() // Int32(topo.xccs_per_die)


@dsl_loc_tracing
def die_tile(tiles, topo):
    """This workgroup's tile: one of its own die's part. Tiles ``[0, tiles/2)`` are
    die 0's, the rest die 1's. Launch a 1-D grid of exactly ``tiles`` workgroups.

    The tile follows from the block index and the XCC: workgroups are dealt to
    XCCs round-robin (from any starting XCC), so every ``num_xccs`` consecutive
    blocks hold each XCC once and a workgroup's rank on its die is
    ``(block // num_xccs) * xccs_per_die + xcc % xccs_per_die``. That needs
    ``tiles`` to be a multiple of ``num_xccs``; a dispatch that deviates from
    round-robin runs some tiles twice and others not at all.
    """
    if tiles % 2 or tiles % topo.num_xccs:
        raise ValueError(f"die_tile: {tiles} tiles must be a multiple of the {topo.num_xccs} XCCs and of 2")
    xpd = topo.xccs_per_die
    rank = (gpu.block_idx.x // topo.num_xccs) * xpd + xcc_id(topo) % Int32(xpd)
    return die_id(topo) * (tiles // 2) + rank
