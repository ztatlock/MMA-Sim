# 2026-04-24 — AMD pairwise oracle bug investigation (plan)

**Status:** open
**Scope:** determine whether the assertion-vs-input-dtype mismatch in
the AMD pairwise path is a genuine upstream bug worth reporting, and
whether our patch faithfully matches the authors' intended semantics.

## What we observed

1. [`mmasim/simulator/arithmetic.py::pairwise_dot`](../mmasim/simulator/arithmetic.py):
   ```python
   def pairwise_dot(a, b, flush_denormal=False):
       assert a.dtype == b.dtype == torch.float32
   ```
2. [`mmasim/simulator/amd.py::mfma.__call__`](../mmasim/simulator/amd.py) operation_type == "pairwise" branch passes `A[i, l:l+gs]`, `B[l:l+gs, j]` directly. For CDNA1/2 f16/bf16 MFMA qualifiers, A/B are source dtype (f16/bf16) after `check_input` enforces dtype.
3. Running the oracle on any CDNA1/2 f16/bf16 MFMA qualifier raises
   `AssertionError` at the first `pairwise_dot` call.
4. Reproduced on fork base commit `c785138`. Not a merge artifact.

## What we patched (locally, committed in our fork)

`mmasim/simulator/amd.py`: add a widening step before the pairwise loop:

```python
if self.operation_type == "pairwise" and A.dtype != torch.float32:
    A = A.to(torch.float32)
    B = B.to(torch.float32)
```

The patch is deliberately narrow: only `pairwise` widens. We verified
that `fused_dot_rd_add` works correctly on narrow dtypes via
`a.double() * b.double()` and needs source-dtype preserved for
`extract_significand_exponent` (min_exp differs by dtype).

With this patch, `fastmma.rust_ref` is bit-exact against the patched
oracle across all 10 affected fixtures.

## What we do not yet know (and must resolve before filing)

1. **When was the assertion added?** Maybe the oracle used to accept
   narrow dtypes and the assertion was added defensively later. Need
   to check `git log -p mmasim/simulator/arithmetic.py` upstream.
2. **Did the paper's test harness hit this path?** The paper says ~1M
   tests per instruction × 10 GPUs. That includes MI100 (CDNA1) and
   MI250X (CDNA2) with f16/bf16. If they really ran these tests, they
   must have had a working oracle for this path. Question: what was
   the oracle's exact state at the time of testing?
3. **Is the oracle's pairwise_dot actually Algorithm 6 (E-FDPA)?**
   Paper's Alg. 6 says:
   ```
   d ← RNE-FP32(c + Σ_{k=0..L-1} a_k b_k)
   ```
   I.e., infinite-precision sum, then single RNE round. The oracle
   computes a binary tree of `fmaf(l, 1.0, r)` — each node rounds to
   f32. These agree only when intermediates don't round (e.g., random
   small values), not in general (e.g., large catastrophic
   cancellation).
4. **Is the oracle's FTZ pairwise actually Algorithm 1?**
   Paper's Alg. 1 defines FTZ-Add and FTZ-Mul as separate elementary
   operations. Alg. 2 (Φ_FTZ-AddMul) then composes them: flush
   subnormals → FTZ-Mul products → pairwise sum of P consecutive
   products → sum partial sums via FTZ-Add, sequentially. Our oracle
   does: `pairwise_dot` (binary tree of fmaf + flush per node) + flush
   after each group sum. Need to verify these agree mathematically.

## Investigation plan

In priority order:

### Step 1: upstream repo forensics (15 min)

- `git log --all -p mmasim/simulator/arithmetic.py | grep -B3 -A3 "assert a.dtype"`
- `git log --all -p mmasim/simulator/amd.py` for the pairwise branch
- `git log --oneline --all mmasim/`
- Check if the assertion was ever absent.

### Step 2: GitHub issues & PRs (5 min)

- `gh issue list --repo microsoft/MMA-Sim --state all`
- `gh pr list --repo microsoft/MMA-Sim --state all`
- Check if anyone else reported this.

### Step 3: look for tests / examples (10 min)

- Is there any test file or example in the repo that exercises the
  AMD pairwise path? Unit tests, demo scripts, notebooks.
- If found, see what inputs it uses. They would either hit the bug
  (strong signal) or go through a different code path (tells us what
  path the authors actually use).

### Step 4: paper algorithm cross-check (30 min)

Open [docs/paper/mma-sim-arxiv-2511.10909.pdf](../docs/paper/mma-sim-arxiv-2511.10909.pdf).
For each of Algorithms 1, 2, 6, 10, 11:

- Transcribe the pseudocode precisely.
- Write the oracle's Python code side-by-side.
- Identify every difference.
- Classify: (a) cosmetic / trivially equivalent, (b) equivalent given
  oracle's caller conventions, (c) different semantics that could
  cause silicon divergence.

Particular suspicion for Alg. 6 (E-FDPA): paper says exact sum +
RNE, oracle does tree of f32 adds. Could diverge on specific inputs.

### Step 5: construct an adversarial test (1 hr, if step 4 turns up doubt)

If step 4 shows the oracle's `pairwise_dot` semantics differ from
Alg. 6, we can construct inputs where:

- Exact sum + RNE (paper) produces result X.
- Tree of f32 adds (oracle) produces result Y ≠ X.

Run the test on both. Confirm they differ.

Caveat: we can't run on silicon to check which one matches. But we
can at least demonstrate the oracle isn't implementing the paper's
algorithm, which is evidence.

### Step 6: reach out (after steps 1-5)

Two possible forms:

**Option A — GitHub issue (no patch):**
Simply describe the reproducer, reference the arXiv page, and ask
"is this the intended behavior? what input path are the paper's tests
using?" No patch attached. Lets authors respond in whatever way is
most useful.

**Option B — GitHub issue + patch:**
Only if we are very confident our widening is exactly right AND the
semantics we implement match silicon. Risk: if authors intended a
different fix, our PR wastes their time.

**Initial lean: Option A.** File a question, let the authors speak.
Patching without their input is presumptuous given we don't have
silicon access.

### Step 7: decide whether to revert our local patch

If authors confirm the widening fix is right: keep it. Document
that we mirror the upstream fix (once merged) or carry a small
divergence.

If authors provide a different fix: update our patch to match.

If authors say "that path was never meant to work, use the API
differently": we may need to refactor our bench/corpus generation
for AMD to use their intended call path.

## Success criteria

We're done investigating when we can answer (with evidence, not
speculation):

1. Does the oracle-as-shipped run the AMD pairwise path? ✗ (known)
2. Is our widening patch lossless relative to the oracle's inner
   math? ✓ (reasoned)
3. Is the oracle's `pairwise_dot` an implementation of the paper's
   Algorithm 6 (E-FDPA), or does it differ?
4. What did the paper's test harness actually run?
5. Is filing an upstream issue net-positive (help the authors) or
   net-negative (noise)?

Only file after (3–5) are answered with evidence.

## Non-goals during investigation

- Don't modify any NVIDIA paths (they're validated per the paper).
- Don't change the Rust kernel for AMD pairwise (matches our patched
  oracle bit-exactly; we'd revalidate if the oracle fix changes).
- Don't reach out to the authors by email / other channels — GitHub
  issue is the standard, auditable venue.
