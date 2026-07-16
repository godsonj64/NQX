# NanoQuant-X Design

## 1. Deployed representation

For a weight matrix \(W\in\mathbb{R}^{m\times n}\), NanoQuant-X deploys

\[
\widehat W
=\operatorname{diag}(s_o)\,U\,\operatorname{diag}(g)\,V^\top
\operatorname{diag}(s_i),
\]

where \(U\in\{-1,+1\}^{m\times r}\),
\(V\in\{-1,+1\}^{n\times r}\), \(s_o\in\mathbb{R}^m\), and
\(s_i\in\mathbb{R}^n\). The strict profile fixes \(g=\mathbf 1\) and does
not store it. The balanced profile stores \(g\in\mathbb{R}^r\) in FP16.

The released initializer reports a continuous product before the binary signs
and boundary scales are applied. NanoQuant-X treats the matrix above as the
only authoritative candidate for selection, validation, and error reporting.

## 2. Weighted objective

Diagonal input and output curvature estimates define the separable objective

\[
\mathcal L
=\sum_{i=1}^{m}\sum_{j=1}^{n}
h^{(o)}_i h^{(i)}_j
\left(W_{ij}-s_{o,i}C_{ij}s_{i,j}\right)^2,
\qquad
C=U\operatorname{diag}(g)V^\top.
\]

With all other variables fixed, each boundary scale has a closed-form weighted
least-squares update. For example,

\[
s_{o,i}
=\frac{\sum_j h^{(i)}_j W_{ij}C_{ij}s_{i,j}}
{\sum_j h^{(i)}_j(C_{ij}s_{i,j})^2+\lambda}.
\]

The input-scale update is symmetric. Alternating these updates is inexpensive,
monotone in the unregularized objective, and optimizes the representation that
will actually execute.

## 3. Per-rank scale solve

For the balanced profile, \(g\) is obtained from a rank-sized linear system.
The normal matrix does not require an \(mn\times r\) design matrix:

\[
H=
\left(U^\top\operatorname{diag}(h^{(o)}\odot s_o^2)U\right)
\odot
\left(V^\top\operatorname{diag}(h^{(i)}\odot s_i^2)V\right).
\]

Its right-hand side is the diagonal of

\[
U^\top\operatorname{diag}(h^{(o)}\odot s_o)
W\operatorname{diag}(h^{(i)}\odot s_i)V.
\]

NanoQuant-X solves the regularized symmetric system with Cholesky and falls
back to least squares if numerical factorization fails.

## 4. Packed storage and exact bit accounting

Binary factors are stored in row-major 32-bit words. Therefore, the actual
factor cost is

\[
B_{UV}=32(m+n)\left\lceil\frac{r}{32}\right\rceil.
\]

With FP16 scales, total layer storage is

\[
B_{\mathrm{strict}}=B_{UV}+16(m+n),
\]

or

\[
B_{\mathrm{balanced}}=B_{UV}+16(m+n+r).
\]

The effective bit rate is \(B/(mn)\). The allocator uses these discrete costs,
including padding, rather than the continuous approximation \(r(m+n)\).

Because the factors are word-packed, all ranks in
\((32(k-1),32k]\) have the same factor cost. The portable path therefore
expands a requested rank to the highest valid rank in its already-paid word.
For example, rank 24 becomes rank 32 without adding strict-profile factor or
scale bits. Balanced mode stores the added rank coefficients explicitly, so
its small metadata increase is still included in BPW.

## 5. Global rank allocation

Each layer begins at the minimum kernel-compatible rank. Additional 32-rank
increments are assigned greedily by estimated curvature-weighted residual
reduction per added storage bit. A diminishing \(1/r\) residual-energy proxy
prevents one layer from absorbing the entire budget. The allocator terminates
before the exact global bit budget would be exceeded.

This policy is deliberately transparent and deterministic. It can later be
replaced by measured per-layer rate-distortion curves without changing the
storage accounting interface.

## 6. Compact global distillation

Caching full teacher logits requires
\(O(NLV)\) values for \(N\) samples, sequence length \(L\), and vocabulary
size \(V\). NanoQuant-X stores the teacher's top-\(k\) indices and probabilities
plus the exact remaining probability mass. The student loss treats the tail as
one aggregate category:

\[
\mathcal L_{\mathrm{KD}}
=-\sum_{t\in\mathrm{top}\text{-}k}p_t\log q_t
-p_{\mathrm{tail}}\log q_{\mathrm{tail}}.
\]

For \(k=128\), this changes cache growth from vocabulary-sized to a fixed
per-token target while preserving total probability. With FP16 probabilities,
int32 indices, and one FP32 tail value, the payload is \(6k+4\) bytes per token,
compared with \(2V\) bytes for BF16 full logits.

## 7. Numerical safeguards

- Curvature diagonals are made finite, positive, mean-normalized, and
  shrinkage-regularized.
- Linear systems are explicitly symmetrized and diagonally regularized.
- Sign extraction maps exact zeros deterministically to \(+1\).
- Negative fitted scales are folded into binary rows or rank columns.
- A non-regression guard retains the original deployed scales if refitting
  increases the weighted objective.
- Reconstruction is chunked over output rows to bound peak workspace.

## 8. Deployment-candidate selection

The ADMM state with the lowest continuous objective is not always the final
state, and neither state is guaranteed to yield the best signs. NanoQuant-X
finalizes both the scheduled final state and the best continuous state, then
selects only by the exact weighted deployed objective. This adds scale/sign
finalization work but no additional ADMM iterations. A non-regression guard
ensures enabling selection cannot make that objective worse than final-only
selection.

## 9. Storage-aware scales

Scale optimization in FP32 can be optimistic when artifacts store FP16 or the
PyTorch module stores BF16. The portable path projects every accepted final
scale to FP16; the production path projects to BF16. It then computes the
optimal scalar gain and tries algebraically equivalent placements in the input,
output, and optional rank scales. Because rounding breaks scale equivalence,
the exact deployed objective decides which placement is retained.

## 10. Prepared portable runtime

For repeated NumPy inference, the input and output scale vectors are fused once
into two floating low-rank factors:

\[
F_i=\operatorname{diag}(s_i)V,\qquad
F_o=\operatorname{diag}(s_o)U\operatorname{diag}(g).
\]

Inference becomes \((XF_i)F_o^\top\), two BLAS operations with no repeated
binary-to-float casts or scale broadcasts. The cache uses
\((m+n)r\) floating values instead of \(mn\) and is optional, lazy, and never
serialized.

## 11. Deterministic safe artifacts

`.nqx` writes canonical ZIP timestamps, permissions, member order, and stored
entries, so identical matrices produce identical bytes. Loading rejects pickle,
encrypted/compressed tensor entries, duplicate or unexpected members, unsafe
paths, dtype/shape mismatches, checksum failures, and declared per-tensor or
aggregate payloads above fixed safety limits.
