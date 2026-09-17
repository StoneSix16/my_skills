#!/usr/bin/env python3
"""Attribute GPU kernel time to NVTX ranges in an nsys capture.

Reads a .nsys-rep (exported on the fly) or an existing .sqlite, and reports:

  - the NVTX range tree, one row per distinct chain of kernel-carrying ranges.
    Keying on the chain rather than the leaf name is what keeps the subtree
    sums additive: a submodule that runs once per stream appears once per
    stream. The range vocabulary is discovered from the capture, not hardcoded,
    so this works on any workload that emits nested NVTX ranges.
  - a self-check: the kernel time attributed across the whole tree must equal
    the capture's total kernel time. If it does not, the attribution rule is
    wrong and every share in the output is suspect.
  - with --idle: how much of each range's window has no kernel running at all.
    The window (union of the range's CPU brackets) answers "was the GPU busy",
    which is a different question from "which range does this kernel belong to".
  - with --gaps: inter-kernel gaps, split by stream, bucketed by the duration
    of the preceding kernel, and classified by the kernel that follows a long
    one. Then, for those long gaps, which two ranges they sit between. The
    first two separate per-kernel dispatch overhead from real stalls; the last
    locates the stall in the code.
  - with --split-by REGEX: kernel time grouped by which matching range is on
    the launch stack, for subdividing one tree node into its streams or passes.

The attribution rule is the one nsys itself uses: a kernel is charged to the
innermost NVTX range that was open on the launching thread at the moment the
runtime API call was made. Choosing a window-based rule instead ("the kernel
started inside the range's CPU bracket") silently drops short ranges, because
the CPU runs ahead of the GPU.

Usage:
    python nsys_attr.py report.sqlite
    python nsys_attr.py report.nsys-rep --idle --gaps
    python nsys_attr.py report.nsys-rep --split-by 'video|action' --top 30
"""

import argparse
import bisect
import collections
import os
import shutil
import subprocess
import sqlite3
import sys
import tempfile


# --------------------------------------------------------------- loading

def open_db(path):
    """Return a sqlite connection, exporting the report first if needed."""
    if path.endswith(".sqlite"):
        return sqlite3.connect(path), None

    tmp = tempfile.mkdtemp(prefix="nsys_attr_")
    out = os.path.join(tmp, "db.sqlite")
    print(f"exporting {path} -> {out}", file=sys.stderr)
    subprocess.run(["nsys", "export", "--type", "sqlite", "-o", out, path],
                   check=True)
    return sqlite3.connect(out), tmp


def load_ranges(cur):
    """Every closed NVTX range: (thread, start, end, name).

    The name lives in `text` for ranges pushed as a string, and in `textId`
    (a key into StringIds) for ranges registered by a framework. Reading only
    `text` drops the second kind entirely, which is usually where the framework
    puts its communication and its own internal phases.
    """
    return cur.execute("""
        SELECT n.globalTid, n.start, n.end,
               COALESCE(n.text, s.value)
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.end IS NOT NULL
    """).fetchall()


def load_kernels(cur):
    """Every kernel with the launch that produced it.

    (gpu_start, gpu_end, launch_thread, launch_time, name, stream)
    """
    return cur.execute("""
        SELECT k.start, k.end, r.globalTid, r.start,
               COALESCE(s.value, '?'), k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
        LEFT JOIN StringIds s ON k.demangledName = s.id
    """).fetchall()


class ThreadRanges:
    """Ranges of one thread, with parent links and a point query.

    Ranges on a thread nest properly (they come from push/pop), so an interval
    either contains another or is disjoint from it. That lets the innermost
    range containing a timestamp be found by locating the latest-starting range
    at or before it and walking up its ancestors -- at most a handful of steps,
    where scanning backwards could walk over every past interval on the thread.
    """

    __slots__ = ("starts", "ends", "names", "parents", "depth")

    def __init__(self, rows):
        # Outer before inner when starts tie, so the parent is pushed first.
        rows = sorted(rows, key=lambda r: (r[0], -r[1]))
        self.starts = [r[0] for r in rows]
        self.ends = [r[1] for r in rows]
        self.names = [r[2] for r in rows]

        self.parents = [-1] * len(rows)
        stack = []
        for i, (start, end, _) in enumerate(rows):
            while stack and self.ends[stack[-1]] <= start:
                stack.pop()
            self.parents[i] = stack[-1] if stack else -1
            stack.append(i)
        self.depth = [0] * len(rows)
        for i in range(len(rows)):
            p = self.parents[i]
            self.depth[i] = 0 if p < 0 else self.depth[p] + 1

    def innermost(self, t):
        """Index of the innermost range containing t, or -1."""
        i = bisect.bisect_right(self.starts, t) - 1
        while i >= 0 and self.ends[i] < t:
            i = self.parents[i]
        return i

    def stack(self, t):
        """Names from the innermost range containing t out to the root."""
        out = []
        i = self.innermost(t)
        while i >= 0:
            out.append(self.names[i])
            i = self.parents[i]
        return out


def build_index(rows):
    by_thread = collections.defaultdict(list)
    for tid, start, end, name in rows:
        by_thread[tid].append((start, end, name or "?"))
    return {tid: ThreadRanges(v) for tid, v in by_thread.items()}


def attribute(kernels, index):
    """Assign each kernel to its innermost range. Returns (assignments, nodes).

    assignments: node key -> [kernel_ms, kernel_count]
    nodes:       node key -> (thread, index into that thread's ranges)
    """
    assignments = collections.defaultdict(lambda: [0.0, 0])
    nodes = {}
    unmatched_ms = 0.0
    for gpu_start, gpu_end, tid, launch_t, name, _stream in kernels:
        ms = (gpu_end - gpu_start) / 1e6
        ranges = index.get(tid)
        i = ranges.innermost(launch_t) if ranges else -1
        if i < 0:
            unmatched_ms += ms
            continue
        key = (tid, ranges.starts[i])
        nodes[key] = (tid, i)
        assignments[key][0] += ms
        assignments[key][1] += 1
    return assignments, nodes, unmatched_ms


# --------------------------------------------------------------- reports

def report_self_check(assignments, kernels, unmatched_ms):
    total = sum(e - s for s, e, *_ in kernels) / 1e6
    charged = sum(v[0] for v in assignments.values())
    print(f"-- attribution self-check --")
    print(f"   kernel total        {total:>12.1f} ms  ({len(kernels)} kernels)")
    print(f"   charged to ranges   {charged:>12.1f} ms")
    print(f"   launched outside    {unmatched_ms:>12.1f} ms")
    ok = abs(charged + unmatched_ms - total) < 1e-6
    print(f"   {'OK' if ok else 'MISMATCH -- shares below are suspect'}")


def _range_path(ranges, i, nodes, tid):
    """Names of the kernel-carrying ranges enclosing `i`, outermost first.

    Ancestors that received no kernels of their own are skipped: they are
    wrappers, and keeping them would add a zero row per level with no
    information. What remains is the chain a reader recognises from the
    capture's own naming ("infer/fwd_*_denoise/attn1").
    """
    chain = []
    while i >= 0:
        if (tid, ranges.starts[i]) in nodes:
            chain.append(ranges.names[i])
        i = ranges.parents[i]
    chain.reverse()
    return tuple(chain)


def report_tree(assignments, nodes, index, top, min_ms, max_depth):
    """Range tree, one row per distinct chain of kernel-carrying ranges.

    Aggregated by the whole ancestor path rather than by the range name alone.
    A name legitimately appears under two parents -- the same submodule runs
    once per stream or per pass -- and collapsing by name would move its time
    into whichever parent happened to be more common, printing a tree whose
    subtree sums do not add up.
    """
    self_ms = collections.Counter()
    kernels = collections.Counter()
    instances = collections.Counter()
    paths = set()
    for key, (tid, i) in nodes.items():
        ms, count = assignments[key]
        path = _range_path(index[tid], i, nodes, tid)
        self_ms[path] += ms
        kernels[path] += count
        instances[path] += 1
        paths.add(path)

    children = collections.defaultdict(list)
    roots = []
    for path in paths:
        if len(path) == 1:
            roots.append(path)
        else:
            children[path[:-1]].append(path)

    totals = {}

    def total_ms(path):
        if path in totals:
            return totals[path]
        totals[path] = 0.0
        totals[path] = self_ms[path] + sum(total_ms(c) for c in children[path])
        return totals[path]

    for path in paths:
        total_ms(path)
    order = sorted(totals, key=lambda p: -totals[p])
    keep = {p for p in order if min_ms == 0 or totals[p] >= min_ms}

    print(f"\n-- range tree (kernel ms, one row per range chain) --")
    print(f"   {'subtree':>9}{'self':>9}{'kern':>9}{'inst':>7}  range")

    def walk(path, indent):
        if path not in keep or indent > max_depth:
            return
        print(f"   {totals[path]:>9.1f}{self_ms[path]:>9.1f}{kernels[path]:>9}"
              f"{instances[path]:>7}  " + "  " * indent + path[-1])
        for child in sorted(children[path], key=lambda p: -totals.get(p, 0)):
            walk(child, indent + 1)

    for path in [p for p in order if p in set(roots)][:top]:
        walk(path, 0)
    if len(totals) - len(keep):
        print(f"   ... {len(totals) - len(keep)} range chain(s) below {min_ms} ms "
              f"hidden (--min-ms)")


def merged(rows):
    out = []
    for s, e in sorted(rows):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def report_idle(nodes, index, kernels, pattern, top):
    """Union window per range vs GPU busy inside it.

    The union matters: instances of one range overlap in time (the CPU runs
    ahead), so summing their spans exceeds the wall clock. A union answers
    "while this range was open, was anything running".
    """
    kernel_union = merged([(s, e) for s, e, *_ in kernels])
    starts = [r[0] for r in kernel_union]

    def busy_in(win):
        tot = 0
        for s, e in win:
            i = bisect.bisect_right(starts, s) - 1
            i = max(i, 0)
            while i < len(kernel_union) and kernel_union[i][0] < e:
                tot += max(0.0, min(e, kernel_union[i][1]) - max(s, kernel_union[i][0]))
                i += 1
        return tot

    by_name = collections.defaultdict(list)
    for tid, i in nodes.values():
        ranges = index[tid]
        name = ranges.names[i]
        if pattern is None or pattern in name:
            by_name[name].append((ranges.starts[i], ranges.ends[i]))

    rows = []
    for name, spans in by_name.items():
        win = merged(spans)
        span = sum(e - s for s, e in win) / 1e6
        busy = busy_in(win) / 1e6
        rows.append((span, name, span, busy, len(spans)))
    rows.sort(reverse=True)
    print(f"\n-- idle inside each range's window (union of its instances) --")
    print(f"   {'window ms':>10}{'busy ms':>9}{'idle ms':>9}{'idle %':>8}{'inst':>7}  range")
    for _, name, span, busy, n in rows[:top]:
        print(f"   {span:>10.1f}{busy:>9.1f}{span - busy:>9.1f}"
              f"{(span - busy) / span * 100:>7.1f}%{n:>7}  {name}")


def attributed_name(index, kernel):
    """Name of the range a kernel is charged to, or a marker if none is open."""
    _s, _e, tid, launch_t, _name, _stream = kernel
    ranges = index.get(tid)
    if not ranges:
        return "<outside any range>"
    i = ranges.innermost(launch_t)
    return ranges.names[i] if i >= 0 else "<outside any range>"


def report_gaps(kernels, index, threshold_ms, top):
    """Inter-kernel gaps, same stream only, bucketed and classified."""
    by_stream = collections.defaultdict(list)
    for k in kernels:
        by_stream[k[5]].append(k)

    print(f"\n-- inter-kernel gaps --")
    print(f"   {'stream':>8}{'kernels':>9}{'span ms':>10}{'busy ms':>10}"
          f"{'gap ms':>10}{'gap %':>8}")
    main = None
    for stream, rows in sorted(by_stream.items(), key=lambda kv: -len(kv[1])):
        rows.sort(key=lambda r: r[0])
        span = rows[-1][1] - rows[0][0]
        busy = sum(r[1] - r[0] for r in rows)
        gap = sum(max(0, rows[i + 1][0] - rows[i][1]) for i in range(len(rows) - 1))
        if main is None:
            main = (stream, rows)
        if len(rows) > 1000:
            print(f"   {stream:>8}{len(rows):>9}{span / 1e6:>10.1f}{busy / 1e6:>10.1f}"
                  f"{gap / 1e6:>10.1f}{gap / span * 100:>7.1f}%")

    stream, rows = main
    dur = [(r[1] - r[0]) / 1e3 for r in rows]
    print(f"\n   kernel durations on stream {stream}: median {sorted(dur)[len(dur) // 2]:.1f} us, "
          f"{sum(1 for d in dur if d < 5) / len(dur) * 100:.0f}% under 5 us")

    # Bucketing by the preceding kernel's duration separates the fixed
    # per-kernel dispatch floor from stalls: a short predecessor cannot hide it.
    buckets = collections.defaultdict(list)
    for i in range(len(rows) - 1):
        gap = rows[i + 1][0] - rows[i][1]
        if gap <= 0:
            continue
        d = (rows[i][1] - rows[i][0]) / 1e3
        label = "<2" if d < 2 else "2-5" if d < 5 else "5-20" if d < 20 \
            else "20-100" if d < 100 else ">100"
        buckets[label].append(gap / 1e3)
    if buckets:
        print(f"\n   gap vs preceding kernel duration (us):")
        for label in ("<2", "2-5", "5-20", "20-100", ">100"):
            if buckets[label]:
                v = sorted(buckets[label])
                print(f"     preceded by {label:>7} us  ->  median gap "
                      f"{v[len(v) // 2]:>7.1f} us   (n={len(v)})")

    # For long gaps only, what runs next identifies what the GPU was waiting on.
    # Timestamps are nanoseconds, so a millisecond threshold scales by 1e6.
    threshold = threshold_ms * 1e6
    successor = collections.Counter()
    count = collections.Counter()
    pair_ms = collections.Counter()
    pair_count = collections.Counter()
    pair_gaps = collections.defaultdict(list)
    total = 0
    for i in range(len(rows) - 1):
        gap = rows[i + 1][0] - rows[i][1]
        if gap <= threshold:
            continue
        total += gap
        cat = classify_kernel(rows[i + 1][4])
        successor[cat] += gap
        count[cat] += 1
        pair = (attributed_name(index, rows[i]), attributed_name(index, rows[i + 1]))
        pair_ms[pair] += gap
        pair_count[pair] += 1
        pair_gaps[pair].append(gap / 1e3)

    if total:
        print(f"\n   gaps over {threshold_ms * 1e3:.0f} us: {total / 1e6:.1f} ms"
              f" total, {sum(count.values())} gaps; what follows them:")
        for cat, ms in successor.most_common(8):
            print(f"     {cat:<38}{ms / 1e6:>9.1f}{ms / total * 100:>7.1f}%"
                  f"{count[cat]:>7}")

        # Where the stall sits, not just what follows it: the range that owns
        # each end of the gap decides which layer has to be changed.
        print(f"\n   the same gaps by the ranges they sit between:")
        print(f"     {'ms':>9}{'n':>7}{'median us':>11}  before -> after")
        for pair, ms in pair_ms.most_common(top):
            v = sorted(pair_gaps[pair])
            before, after = pair
            label = f"{before} -> {after}" if before != after \
                else f"{before} (internal)"
            print(f"     {ms / 1e6:>9.1f}{pair_count[pair]:>7}"
                  f"{v[len(v) // 2]:>11.1f}  {label}")


def classify_kernel(name):
    """Coarse family of a kernel, by the library that emitted it."""
    lowered = name.lower()
    for label, needles in (
            ("collective / shard transfer", ("allgather", "all_gather", "nccl",
                                             "multi_tensor_apply",
                                             "split_with_sizes", "reducescatter")),
            ("gemm", ("gemm", "cutlass", "cublas")),
            ("attention", ("flash", "attention", "sdpa", "fmha", "softmax")),
            ("memcpy", ("memcpy", "direct_copy", "_copy")),
            ("norm", ("_norm",)),
            ("index / gather", ("index_", "gather", "nonzero")),
            ("reduce", ("reduce_kernel",)),
            ("elementwise", ("elementwise", "vectorized", "multi_tensor")),
    ):
        if any(n in lowered for n in needles):
            return label
    return "other"


def report_split(kernels, index, pattern, top):
    """Kernel time split by which matching range is on the launch stack.

    Needed whenever one tree node covers more than one execution path -- two
    streams, two passes -- because NVTX range names are flat and the
    distinguishing range sits in the middle of the stack, not at the innermost
    level.
    """
    import re
    rx = re.compile(pattern)
    groups = collections.Counter()
    calls = collections.Counter()
    for gpu_start, gpu_end, tid, launch_t, name, _stream in kernels:
        ranges = index.get(tid)
        if not ranges:
            continue
        hit = next((n for n in ranges.stack(launch_t) if rx.search(n)), None)
        if hit is None:
            continue
        groups[hit] += (gpu_end - gpu_start) / 1e6
        calls[hit] += 1
    if not groups:
        print(f"\n   no range matched {pattern!r}")
        return
    print(f"\n-- kernel time by stack member matching /{pattern}/ --")
    print(f"   {'ms':>10}{'kernels':>10}  range")
    for name, ms in groups.most_common(top):
        print(f"   {ms:>10.1f}{calls[name]:>10}  {name}")


# --------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Attribute GPU kernel time to NVTX ranges in an nsys "
                    "capture, and analyse where the GPU was idle.")
    parser.add_argument("report", help=".nsys-rep or exported .sqlite")
    parser.add_argument("--top", type=int, default=20,
                        help="rows to print per table (default 20)")
    parser.add_argument("--min-ms", type=float, default=0.0,
                        help="hide tree nodes below this total kernel time")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="tree depth to print (default 6)")
    parser.add_argument("--idle", action="store_true",
                        help="add the per-range window/busy/idle table")
    parser.add_argument("--idle-match", metavar="SUBSTR",
                        help="restrict --idle to range names containing SUBSTR")
    parser.add_argument("--gaps", action="store_true",
                        help="add the inter-kernel gap analysis")
    parser.add_argument("--gap-threshold-ms", type=float, default=0.2,
                        help="a gap above this is treated as a stall (default 0.2)")
    parser.add_argument("--split-by", metavar="REGEX",
                        help="also split kernel time by the matching range that "
                             "is on the launch stack")
    args = parser.parse_args()

    db, tmp = open_db(args.report)
    try:
        cur = db.cursor()
        index = build_index(load_ranges(cur))
        kernels = load_kernels(cur)
        assignments, nodes, unmatched = attribute(kernels, index)

        report_self_check(assignments, kernels, unmatched)
        report_tree(assignments, nodes, index, args.top, args.min_ms,
                    args.max_depth)
        if args.split_by:
            report_split(kernels, index, args.split_by, args.top)
        if args.idle:
            report_idle(nodes, index, kernels, args.idle_match, args.top)
        if args.gaps:
            report_gaps(kernels, index, args.gap_threshold_ms, args.top)
    finally:
        db.close()
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
