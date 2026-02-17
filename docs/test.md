RoPE
    ═══════════════════════════════════════════════════════
    (Rotary Position Embeddings)

    Core Idea: Encode position as ROTATION angle
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │     Position 0      Position 1      Position 2      │
    │         │               │               │           │
    │         ▼               ▼               ▼           │
    │       ─────►         ───►           ──►             │
    │      (θ = 0°)      (θ = 30°)      (θ = 60°)         │
    │                                                     │
    │     Each position rotates the vector more           │
    └─────────────────────────────────────────────────────┘


    2D Rotation Applied to Embedding Pairs:
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Embedding: [x₁, x₂, x₃, x₄, ..., xₐ]              │
    │               ↓   ↓   ↓   ↓                         │
    │              [pair₁] [pair₂]  ...                   │
    │                                                     │
    │   Rotation Matrix for position m:                   │
    │                                                     │
    │   ┌                      ┐                          │
    │   │  cos(mθ)   -sin(mθ)  │                          │
    │   │  sin(mθ)    cos(mθ)  │                          │
    │   └                      ┘                          │
    │                                                     │
    │   Apply to each pair of dimensions                  │
    └─────────────────────────────────────────────────────┘


    Why Rotation Works for Relative Position:
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   Query at position m:  Qₘ = Rotate(q, mθ)            │
    │   Key at position n:    Kₙ = Rotate(k, nθ)             │
    │                                                       │
    │   Attention: Qₘ · Kₙ depends on (m - n)                │
    │                                                     │
    │   ┌───────────────────────────────────────┐         │
    │   │ The dot product naturally encodes     │         │
    │   │ RELATIVE position (m - n), not        │         │
    │   │ absolute positions m and n!           │         │
    │   └───────────────────────────────────────┘         │
    │                                                     │
    └─────────────────────────────────────────────────────┘


    Benefits of RoPE:
    ┌─────────────────────────────────────────────────────┐
    │ ✓ No learned parameters - purely mathematical       │
    │ ✓ Naturally encodes relative positions              │
    │ ✓ Extrapolates to longer sequences than trained     │
    │ ✓ Efficient - applied directly to Q and K           │
    └─────────────────────────────────────────────────────┘