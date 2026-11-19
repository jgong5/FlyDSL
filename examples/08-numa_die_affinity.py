# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Die-affine decode GEMV (y = W @ x) with `flydsl.expr.rocdl.numa`.

Rows [0, N/2) of W live on die 0 and the rest on die 1 (`pack_die_split`), and
the `@flyc.jit` wrapper views W through `die_split_tensor`, so every `fx.copy`
lands on the right die whatever the tile size. The kernel body is the plain
GEMV's; only its tile index comes from `die_tile`.

On gfx942/gfx95x this runs the gfx1250 topology emulated: it checks results and
that every element a tile reads lies on its die (`die_of_address`), not
die-local bandwidth.
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.rocdl import numa


def build(N, K, BN, BK, topo=None):
    TPR = BK // 4  # threads per row: 4 fp32 per 128-bit load
    assert BN * TPR == 256 and TPR <= 64 and N % BN == 0 and K % BK == 0
    NT = N // BN

    @flyc.kernel
    def gemv_kernel(W: fx.Tensor, X: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        tile_n = fx.block_idx.x if fx.const_expr(topo is None) else numa.die_tile(NT, topo)

        W = fx.rocdl.make_buffer_tensor(W)
        X = fx.rocdl.make_buffer_tensor(X)
        gW = fx.flat_divide(W, (BN, BK))[None, None, tile_n, None]  # (BN, BK, K/BK)
        gX = fx.flat_divide(X, (BN, BK))[None, None, 0, None]

        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        tile_mn, tv = fx.make_layout_tv(fx.make_layout((BN, TPR), (TPR, 1)), fx.make_layout((1, 4), (1, 1)))
        thr = fx.make_tiled_copy(atom, tv, tile_mn).get_slice(tid)
        tW = thr.partition_S(gW)
        tX = thr.partition_S(gX)
        rW = fx.make_fragment_like(tW[None, None, None, 0])
        rX = fx.make_fragment_like(tX[None, None, None, 0])
        acc = fx.make_fragment_like(rW)
        acc.fill(0)
        for kt in range(K // BK):
            fx.copy(atom, tW[None, None, None, kt], rW)
            fx.copy(atom, tX[None, None, None, kt], rX)
            acc.store(acc.load() + rW.load() * rX.load())

        s = acc.load().reduce(fx.ReductionOp.ADD)
        for i in fx.range_constexpr(TPR.bit_length() - 1):
            s = s + s.shuffle_xor(TPR >> (i + 1), 64)
        if tid % TPR == 0:
            fx.ptr_store(s, fx.add_offset(fx.get_iter(Y), tile_n * BN + tid // TPR))

    @flyc.jit
    def gemv(W: fx.Tensor, X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        rows = fx.make_layout((N, K), (K, 1))
        if fx.const_expr(topo is None):
            W = fx.Tensor(fx.make_view(fx.get_iter(W), rows))
        else:
            W = numa.die_split_tensor(W, rows, N // 2 * K, topo)
        X = fx.Tensor(fx.make_view(fx.get_iter(X), fx.make_layout((N, K), (0, 1))))  # x broadcast to rows
        gemv_kernel(W, X, Y).launch(grid=(NT, 1, 1), block=(256, 1, 1), stream=stream)

    return gemv


def check(N, K, BN, BK, topo):
    w = torch.randn(N, K, device="cuda")
    x = torch.randn(K, device="cuda")
    ref = w @ x
    split = N // 2 * K

    y0 = torch.zeros(N, device="cuda")
    build(N, K, BN, BK)(w, x, y0)
    wp = numa.pack_die_split(w, split, topo)
    y1 = torch.zeros(N, device="cuda")
    build(N, K, BN, BK, topo)(wp, x, y1)
    torch.cuda.synchronize()

    g = torch.arange(N * K, device="cuda")
    phys = wp.data_ptr() + numa.die_split_offsets(g, split, 4, topo) * 4
    on_its_die = torch.equal(numa.die_of_address(phys, topo), g // split)
    tol = dict(rtol=1e-4, atol=1e-3)
    ok = torch.allclose(y0, ref, **tol) and torch.allclose(y1, ref, **tol) and on_its_die
    print(f"N={N:5d} K={K:5d} tile={BN:2d}x{BK:3d}: correct {ok}")
    return ok, w, wp, x


def bench(N, K, BN, BK, topo, iters=200):
    _, w, wp, x = check(N, K, BN, BK, topo)
    y = torch.zeros(N, device="cuda")
    stream = torch.cuda.current_stream()
    for name, fn, W in (("plain", build(N, K, BN, BK), w), ("numa ", build(N, K, BN, BK, topo), wp)):
        compiled = flyc.compile(fn, W, x, y, stream)  # a @flyc.jit call re-traces every time
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for i in range(iters + 10):
            if i == 10:
                e0.record()
            compiled(W, x, y, stream)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) / iters * 1e3
        print(f"  {name}: {us:7.1f} us  {N * K * 4 / us / 1e3:6.0f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", action="store_true", help="also time plain vs. die-affine")
    args = parser.parse_args()
    topo = numa.topology(emulate="gfx1250")
    ok = True
    # Tiles from 64x16 to 4x256; K=7168 rows (28 KiB) cross 16 KiB chunks mid-row.
    for shape in ((4096, 7168, 16, 64), (4096, 7168, 4, 256), (2048, 4096, 64, 16)):
        ok &= check(*shape, topo)[0]
    print(f"All passed: {ok}")
    if args.bench:
        bench(16384, 7168, 8, 128, topo)
