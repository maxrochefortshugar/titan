# Incidents

Kernel panics and freezes on the serving machine, with what was running and what changed afterwards. Both entries so far are the same panic: `watchdog timeout: no checkins from watchdogd in 90 seconds`, panicked task `kernel_task`, identical kernel backtrace offsets, CPUs otherwise idle (only WindowServer in the thread table), a `python3.11` MLX process the only heavy work. Swap and the compressor were healthy in the second snapshot, so neither was memory exhaustion. That signature is a GPU-side stall that the kernel could not recover from: a Metal command the firmware never finished, or a GPU page-fault storm on memory the driver could not service in time.

## 2026-09-12 00:23, oMLX overlay, resident n-gram table

The 32 GB packed n-gram table was mapped resident and read by the lookup kernel on the GPU. Two UI freezes preceded the panic. Rule since then: the table is streamed by rows, never resident, and Titan's reader refuses any resident or preload mode at open time.

## 2026-09-12 23:04, Titan, instrumented per-layer breakdown at 64k

`bench/decode/layer_breakdown.py --contexts 600,64000 --widths 1,4 --repeats 12` on the real checkpoint, run directly (not through the server) in the background by an agent. The script wraps every layer type's `__call__` and, in its `run` pass, evaluates each scope's own output and calls `mx.synchronize()` per scope, then rewinds the verify block after every step: thousands of tiny command buffers with forced evaluation of intermediates at 64k. The tree was at f07e1d5 with no model edits; the same committed code had served 64k, width 4 requests for over an hour through `titan serve` without incident. The watchdog fired about 90 seconds after the agent's last recorded action; its log lived in /tmp and did not survive the reboot.

What is different from the ordinary path: forced evaluation of every intermediate (including the sparse-attention gather and the expert gather at width 4) as separate command buffers, per-scope device syncs, and repeated state rewinds. The fast kernels were on (default); the same kernels ran fine under the server.

## Rules

1. Real-model measurements go through `titan serve` and its profiler and `/metrics`, or through `bench/decode/run_round2.sh`. No instrumented in-process harness that forces per-scope evaluation or per-scope `mx.synchronize()` on the real checkpoint. Layer-level attribution is measured on the synthetic configuration, or on the real model with at most one sync per step.
2. First real-model run of any new kernel path happens with `kernels.reference_only = true` as the control, then with only that kernel enabled, at short context before 64k.
3. Every custom Metal kernel has a bounds test at the largest real shape it can see (64k context, verify width 8, top-k 10, 512 experts) on synthetic data before it is enabled by default.
4. One model process at a time, 110 GB hard limit, the per-port lock and headroom wait in `titan serve` (see `titan/observability/memory_guard.py`), and the memory watch in the operator session. These were all in place and did not fire: this panic was not a memory event.
5. Anything that must survive a crash is written under the repo or `~/inference-server`, never under /tmp.
