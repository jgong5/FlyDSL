# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for flydsl.expr.rocdl.numa: die-split placement and die-affine tiles (gfx1250, emulated)."""

import dataclasses

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.rocdl import numa

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.rocm_lower]

GFX1250 = numa.Topology("gfx1250", num_xccs=8, emulated=True)  # host-side checks only


def _device_arch():
    if torch is None or not torch.cuda.is_available():
        return ""
    from flydsl.runtime.device import get_rocm_arch

    try:
        return str(get_rocm_arch())
    except RuntimeError:
        return ""


device = pytest.mark.skipif(
    not _device_arch().startswith(("gfx942", "gfx95")), reason="emulation needs a gfx942/gfx95x GPU"
)


# ---------------------------------------------------------------- host


@pytest.mark.l1b_target_dialect
@pytest.mark.parametrize("elem_bytes", [1, 2, 4])
def test_die_split_offsets_place_each_part_on_its_die(elem_bytes):
    chunk = GFX1250.chunk_bytes // elem_bytes
    split = 3 * chunk  # parts of unequal size: 3 chunks, then 5
    g = list(range(8 * chunk))
    phys = [numa.die_split_offsets(i, split, elem_bytes, GFX1250) for i in g]
    assert len(set(phys)) == len(phys)
    assert [numa.die_of_address(p * elem_bytes, GFX1250) for p in phys] == [int(i >= split) for i in g]
    # Within a chunk the offsets stay contiguous.
    assert all(phys[i + 1] == phys[i] + 1 for i in range(len(g) - 1) if (i + 1) % chunk)


@pytest.mark.l1b_target_dialect
def test_die_split_offsets_cover_every_color_bit():
    # Chunk indices up to 2**13 exercise all of address bits 14..26.
    split = GFX1250.chunk_bytes * 4096  # in bytes, elem_bytes=1
    for chunk in range(0, 2 * 4096, 37):
        g = chunk * GFX1250.chunk_bytes
        assert numa.die_of_address(numa.die_split_offsets(g, split, 1, GFX1250), GFX1250) == int(g >= split)


@pytest.mark.l1b_target_dialect
def test_die_split_rejects_unaligned_split():
    with pytest.raises(ValueError, match="multiple of"):
        numa.die_split_offsets(0, 100, 4, GFX1250)


@pytest.mark.l1b_target_dialect
def test_real_gfx1250_interfaces_are_stubs():
    real = dataclasses.replace(GFX1250, emulated=False)
    with pytest.raises(NotImplementedError, match="not available yet"):
        numa.empty(16, torch.float32 if torch else None, "cpu", real)


# ---------------------------------------------------------------- device

N, K = 64, 7168  # 28 KiB rows cross 16 KiB chunks mid-row
SPLIT = N // 2 * K


def _copy_through_view(BN, BK, topo):
    TPR = BK // 4

    @flyc.kernel
    def copy_kernel(W: fx.Tensor, Out: fx.Tensor):
        W = fx.rocdl.make_buffer_tensor(W)
        Out = fx.rocdl.make_buffer_tensor(Out)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        tile_mn, tv = fx.make_layout_tv(fx.make_layout((BN, TPR), (TPR, 1)), fx.make_layout((1, 4), (1, 1)))
        thr = fx.make_tiled_copy(atom, tv, tile_mn).get_slice(fx.thread_idx.x)
        tW = thr.partition_S(fx.flat_divide(W, (BN, BK))[None, None, fx.block_idx.x, None])
        tO = thr.partition_D(fx.flat_divide(Out, (BN, BK))[None, None, fx.block_idx.x, None])
        frag = fx.make_fragment_like(tW[None, None, None, 0])
        for kt in range(K // BK):
            fx.copy(atom, tW[None, None, None, kt], frag)
            fx.copy(atom, frag, tO[None, None, None, kt])

    @flyc.jit
    def run(W: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        W = numa.die_split_tensor(W, fx.make_layout((N, K), (K, 1)), SPLIT, topo)
        Out = fx.Tensor(fx.make_view(fx.get_iter(Out), fx.make_layout((N, K), (K, 1))))
        copy_kernel(W, Out).launch(grid=(N // BN, 1, 1), block=(BN * TPR, 1, 1), stream=stream)

    return run


@pytest.mark.l2_device
@device
@pytest.mark.parametrize("BN, BK", [(16, 64), (4, 256), (64, 16)])
def test_die_split_tensor_reads_logical_values(BN, BK):
    topo = numa.topology(emulate="gfx1250")
    w = torch.randn(N, K, device="cuda")
    wp = numa.pack_die_split(w, SPLIT, topo)
    assert wp.data_ptr() % topo.base_align == 0
    out = torch.zeros_like(w)
    _copy_through_view(BN, BK, topo)(wp, out)
    torch.cuda.synchronize()
    assert torch.equal(out, w)


def _tile_recorder(tiles, topo):
    @flyc.kernel
    def record_kernel(hits: fx.Tensor, dies: fx.Tensor):
        tile = numa.die_tile(tiles, topo)
        if fx.thread_idx.x == 0:
            fx.atomic_add(fx.add_offset(fx.get_iter(hits), tile), fx.Int32(1))
            fx.ptr_store(numa.die_id(topo), fx.add_offset(fx.get_iter(dies), tile))

    @flyc.jit
    def run(hits: fx.Tensor, dies: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        record_kernel(hits, dies).launch(grid=(tiles, 1, 1), block=(64, 1, 1), stream=stream)

    return run


@pytest.mark.l2_device
@device
def test_die_tile_runs_each_tile_once_on_its_die():
    # Also checks the round-robin dispatch the static mapping assumes.
    topo = numa.topology(emulate="gfx1250")
    tiles = 125 * topo.num_xccs
    hits = torch.zeros(tiles, dtype=torch.int32, device="cuda")
    dies = torch.full_like(hits, -1)
    _tile_recorder(tiles, topo)(hits, dies)
    torch.cuda.synchronize()
    assert torch.equal(hits, torch.ones_like(hits))
    assert torch.equal(dies.long(), (torch.arange(tiles, device="cuda") >= tiles // 2).long())


@pytest.mark.l2_device
@device
def test_topology_needs_explicit_emulation():
    with pytest.raises(NotImplementedError, match="emulate"):
        numa.topology()
    assert numa.topology(emulate="gfx1250").emulated
