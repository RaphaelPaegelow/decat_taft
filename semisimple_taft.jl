# Code used to ease computations in the decategorification/categorification of module categories of the semisimplification of the Taft category
# This code has been partly AI generated

using Oscar
using SparseArrays

# ============================================================================
# Taft algebra
# ============================================================================

"""
    TaftAlgebra

A wrapper around the OSCAR realization of

    T(n) = K<g,z> / (g^n - 1, z^n, g*z - q*z*g),

where `q` is a primitive `n`-th root of unity.

The field `coproduct` records the convention used for tensor products:

* `:zid_gz` means Delta(z) = z  tensor id + g tensor z;
* `:idz_zg` means Delta(z) = id tensor z  + z tensor g.
"""
struct TaftAlgebra{KType, QType, AType, EType}
    n::Int
    K::KType
    q::QType
    A::AType
    g::EType
    z::EType
    coproduct::Symbol
end

const TAFT_CACHE = Dict{Tuple{Int, Symbol}, Any}()

"""
    taft_algebra(n::Int; coproduct=:zid_gz)

Construct (and cache) the Taft algebra T(n).
"""
function taft_algebra(n::Int; coproduct::Symbol=:zid_gz)
    n >= 2 || throw(DomainError(n, "n must be at least 2"))
    coproduct in (:zid_gz, :idz_zg) ||
        throw(ArgumentError("coproduct must be :zid_gz or :idz_zg"))

    key = (n, coproduct)
    return get!(TAFT_CACHE, key) do
        K, q = cyclotomic_field(n, "q")

        R, (z, g) = polynomial_ring(K, [:z, :g])

        # OSCAR PBW relations have the form x_j*x_i = relation[i,j].
        # With (x_1, x_2) = (z, g), this is g*z = q*z*g.
        rel = strictly_upper_triangular_matrix([q*z*g])
        A, (zA, gA) = pbw_algebra(R, rel, deglex(gens(R)))

        I = two_sided_ideal(A, [zA^n, gA^n - one(A)])
        Q, pi = quo(A, I)

        T = TaftAlgebra(n, K, q, Q, pi(gA), pi(zA), coproduct)

        # Validate each newly created quotient algebra once (inside the cache).
        @assert T.g^n == one(T.A)
        @assert T.z^n == zero(T.A)
        @assert T.g*T.z == T.q*T.z*T.g
        return T
    end
end


# ============================================================================
# Indecomposable (non-projective) Taft modules
# ============================================================================

abstract type AbstractTaftRepresentation end

"""
    M(l, r, T)
    M(l, r, n=6)

The indecomposable non-projective module M(l,r) over a Taft algebra T(n).
"""
struct M{TType} <: AbstractTaftRepresentation
    l::Int                              # l of M_{l,r}
    r::Int                              # r of M_{l,r}
    T::TType                            # parent Taft algebra
end

function M(l::Int, r::Int, T::TaftAlgebra)
    n = T.n
    if l < 1 || l >= n
        throw(DomainError(l, "l must be between 1 and $(n-1)"))
    end
    if r < 0 || r >= n
        throw(DomainError(r, "r must be between 0 and $(n-1)"))
    end
    return M{typeof(T)}(l, r, T)
end

# Compatibility constructor: existing calls M(l,r,n) continue to work.
M(l::Int, r::Int, n::Int=6) = M(l, r, taft_algebra(n))

taft_algebra(m::M) = m.T
taft_degree(m::M) = m.T.n

# Compatibility for existing external code that may still use m.n.
function Base.getproperty(m::M, s::Symbol)
    if s === :n
        return getfield(getfield(m, :T), :n)
    end
    return getfield(m, s)
end

Base.propertynames(::M, private::Bool=false) = (:l, :r, :T, :n)


# ============================================================================
# Concrete representation data for M(l,r)
# ============================================================================

# The cache is keyed first by the actual OSCAR quotient-algebra parent, so two
# unrelated parent objects are never accidentally mixed even if they have the
# same value of n.
const MODULE_REPRESENTATION_CACHE = IdDict{Any, Dict{Tuple{Int, Int}, Any}}()

"""
    representation_data(m::M)

Return a named tuple `(V=..., G=..., Z=...)` containing the underlying vector
space and the matrices of `g` and `z`.

Basis convention used here:

    g*v_i = q^(r+i-1) v_i,
    z*v_i = v_(i+1) for i < l,
    z*v_l = 0.

Internally, action matrices use the column-vector convention: basis vectors are
columns and a linear map acts by `v -> A*v`. Thus the relation `g*z = q*z*g`
is represented by `G*Z = q*Z*G`. OSCAR's `ModuleHomomorphism` constructor uses
row-coordinate matrices, so we transpose only when converting these matrices to
OSCAR homomorphisms.
"""
function representation_data(m::M)
    T = taft_algebra(m)
    reps_for_T = get!(MODULE_REPRESENTATION_CACHE, T.A) do
        Dict{Tuple{Int, Int}, Any}()
    end

    key = (m.l, m.r)
    return get!(reps_for_T, key) do
        K = T.K
        q = T.q
        l = m.l

        V = vector_space(K, l)
        G = zero_matrix(K, l, l)
        Z = zero_matrix(K, l, l)

        for i in 1:l
            G[i, i] = q^mod(m.r + i - 1, T.n)
            if i < l
                Z[i + 1, i] = one(K)
            end
        end

        @assert G*Z == q*(Z*G)
        (V=V, G=G, Z=Z)
    end
end

underlying_space(m::M) = representation_data(m).V
g_action_matrix(m::M) = representation_data(m).G
z_action_matrix(m::M) = representation_data(m).Z
basis(m::M) = basis(underlying_space(m))

# OSCAR's ModuleHomomorphism constructor uses row-coordinate matrices.
# Our internal matrices use columns, so transpose at this boundary only.
g_action(m::M) = ModuleHomomorphism(
    underlying_space(m),
    underlying_space(m),
    transpose(g_action_matrix(m)),
)
z_action(m::M) = ModuleHomomorphism(
    underlying_space(m),
    underlying_space(m),
    transpose(z_action_matrix(m)),
)


# ============================================================================
# Tensor-product representations and kernel computations
# ============================================================================

struct TaftTensorProduct{LType, RType, VType, MType} <: AbstractTaftRepresentation
    left::LType
    right::RType
    V::VType
    G::MType
    Z::MType
end

taft_algebra(P::TaftTensorProduct) = taft_algebra(P.left)
taft_degree(P::TaftTensorProduct) = taft_degree(P.left)
underlying_space(P::TaftTensorProduct) = P.V
g_action_matrix(P::TaftTensorProduct) = P.G
z_action_matrix(P::TaftTensorProduct) = P.Z
basis(P::TaftTensorProduct) = basis(P.V)

function _identity_matrix(K, d::Int)
    I = zero_matrix(K, d, d)
    for i in 1:d
        I[i, i] = one(K)
    end
    return I
end

# Matrix of A tensor B in the lexicographic basis
# (e_1 tensor f_1, ..., e_1 tensor f_m, e_2 tensor f_1, ...).
#
# Column-vector convention: if A*e_i = sum_u A[u,i] e_u and
# B*f_j = sum_v B[v,j] f_v, then
#
#   (A tensor B)(e_i tensor f_j)
#       = sum_{u,v} A[u,i] B[v,j] (e_u tensor f_v).
function _tensor_action_matrix(A, B, K)
    a_rows, a_cols = nrows(A), ncols(A)
    b_rows, b_cols = nrows(B), ncols(B)
    C = zero_matrix(K, a_rows*b_rows, a_cols*b_cols)

    src(i, j) = (i - 1)*b_cols + j
    dst(u, v) = (u - 1)*b_rows + v

    for i in 1:a_cols, j in 1:b_cols
        s = src(i, j)
        for u in 1:a_rows
            a = A[u, i]
            iszero(a) && continue
            for v in 1:b_rows
                b = B[v, j]
                iszero(b) && continue
                C[dst(u, v), s] += a*b
            end
        end
    end
    return C
end

"""
    taft_tensor_product(X::M, Y::M)

Construct the concrete representation X tensor Y using the coproduct stored in
`X.T`. The underlying basis is lexicographic:

    v_1 tensor w_1, ..., v_1 tensor w_m,
    v_2 tensor w_1, ..., v_l tensor w_m.
"""
function taft_tensor_product(X::M, Y::M)
    X.T.A === Y.T.A ||
        throw(ArgumentError("X and Y must be modules over the same Taft algebra parent"))

    T = taft_algebra(X)
    K = T.K

    GX = g_action_matrix(X)
    GY = g_action_matrix(Y)
    ZX = z_action_matrix(X)
    ZY = z_action_matrix(Y)

    IX = _identity_matrix(K, X.l)
    IY = _identity_matrix(K, Y.l)

    G = _tensor_action_matrix(GX, GY, K)

    if T.coproduct == :zid_gz
        # Delta(z) = z tensor 1 + g tensor z
        Z = _tensor_action_matrix(ZX, IY, K) +
            _tensor_action_matrix(GX, ZY, K)
    elseif T.coproduct == :idz_zg
        # Delta(z) = 1 tensor z + z tensor g
        Z = _tensor_action_matrix(IX, ZY, K) +
            _tensor_action_matrix(ZX, GY, K)
    else
        error("unsupported coproduct convention $(T.coproduct)")
    end

    V = vector_space(K, X.l*Y.l)
    P = TaftTensorProduct(X, Y, V, G, Z)

    @assert G*Z == T.q*(Z*G)
    return P
end

g_action(P::TaftTensorProduct) = ModuleHomomorphism(P.V, P.V, transpose(P.G))
z_action(P::TaftTensorProduct) = ModuleHomomorphism(P.V, P.V, transpose(P.Z))

"""
    z_power_action(X, k)

Return the linear map giving the action of z^k on a concrete Taft
representation (`M` or `TaftTensorProduct`).
"""
function z_power_action(X::AbstractTaftRepresentation, k::Int)
    k >= 0 || throw(DomainError(k, "k must be nonnegative"))
    V = underlying_space(X)
    return ModuleHomomorphism(V, V, transpose(z_action_matrix(X)^k))
end

"""
    z_kernel(X, k)

Return `(K, inc)` where `K = ker(z^k)` as an actual OSCAR subspace and `inc`
is its canonical inclusion into the underlying vector space.
"""
z_kernel(X::AbstractTaftRepresentation, k::Int) = kernel(z_power_action(X, k))

"""
    z_kernel_on_subspace(X, B, k=9)

Let the columns of `B` be a basis (or, more generally, a spanning family) of a
subspace of `underlying_space(X)`. Compute the kernel of the composition

    K^d --B--> X --z^k--> X,

where `d = ncols(B)`.

All representation matrices in this file use the column-vector convention, so
the matrix of the composition is

    C = Z^k * B,

where `Z = z_action_matrix(X)`. The return value is a matrix `K0` whose columns
form a basis of the coefficient kernel:

    C * K0 = 0.

Thus each column `c` of `K0` gives a vector `B*c` in the chosen subspace which
is killed by `z^k`. If actual ambient vectors are wanted, they are the columns
of `B*K0`.
"""
function z_kernel_on_subspace(
    X::AbstractTaftRepresentation,
    B,
    k::Int=9,
)
    k >= 0 || throw(DomainError(k, "k must be nonnegative"))

    V = underlying_space(X)
    nrows(B) == dim(V) ||
        throw(DimensionMismatch(
            "B must have $(dim(V)) rows, one for each basis vector of the representation"
        ))

    # Column-vector convention:
    #
    #     c -> B*c -> Z^k*B*c.
    #
    C = z_action_matrix(X)^k * B

    # Right kernel: the columns of K0 satisfy C*K0 = 0.
    return kernel(C; side=:right)
end

"""
    z_kernel_ambient_span(X, B, k=9)

Return the *ambient* spanning matrix `B * z_kernel_on_subspace(X, B, k)`.
If `B` is not injective, these columns need not be independent (and may be zero).
"""
function z_kernel_ambient_span(X::AbstractTaftRepresentation, B, k::Int=9)
    return B * z_kernel_on_subspace(X, B, k)
end

function tensor_index(P::TaftTensorProduct, i::Int, j::Int)
    l = P.left.l
    m = P.right.l
    1 <= i <= l || throw(BoundsError())
    1 <= j <= m || throw(BoundsError())
    return (i - 1)*m + j
end

tensor_basis_vector(P::TaftTensorProduct, i::Int, j::Int) =
    basis(P)[tensor_index(P, i, j)]

function compute_all_u(a::Int, k::Int)
    0 <= a <= 6 || throw(DomainError(a, "compute_all_u requires 0 <= a <= 6"))
    k >= 0 || throw(DomainError(k, "k must be nonnegative"))
    m     = M(7,9,12)
    Mtens = taft_tensor_product(m,m)
    K     = m.T.K
    d     = 7-a
    B     = zero_matrix(K,7*7,d)
    for i in 1:d
       B[tensor_index(Mtens,i,d+1-i),i] = one(K)
    end
    return z_kernel_on_subspace(Mtens,B,k)
end

function compute_all_v(a::Int, k::Int)
    0 <= a <= 5 || throw(DomainError(a, "compute_all_v requires 0 <= a <= 5"))
    k >= 0 || throw(DomainError(k, "k must be nonnegative"))
    m     = M(7,9,12)
    wa    = M(2*a+1, mod(-a,12),12)
    Mtens = taft_tensor_product(wa,m)
    K     = m.T.K
    d     = a+1
    B     = zero_matrix(K,7*(2*a+1),d)
    for i in 1:d
       B[tensor_index(Mtens,i,a+2-i),i] = one(K)
    end
    return z_kernel_on_subspace(Mtens,B,k)
end

# ============================================================================
# Formal decategorification data
# ============================================================================

struct MExpr
    elem::FinGenAbGroupElem
    n::Int
end

taft_degree(me::MExpr) = me.n

const DECATEGORIFICATION_CACHE = Dict{Int, NamedTuple{(:group, :to_oscar, :to_struct), Tuple{FinGenAbGroup, Dict{Tuple{Int,Int}, FinGenAbGroupElem}, Dict{FinGenAbGroupElem, Tuple{Int,Int}}}}}()
const CACHE = DECATEGORIFICATION_CACHE  # backwards compatibility

function get_env(n::Int)
    if !haskey(DECATEGORIFICATION_CACHE, n)
        l_range, r_range = 1:n-1, 0:n-1
        F = free_abelian_group(length(l_range) * length(r_range))
        g = gens(F)

        to_oscar = Dict{Tuple{Int,Int}, FinGenAbGroupElem}()
        to_struct = Dict{FinGenAbGroupElem, Tuple{Int,Int}}()
        idx = 1
        for li in l_range, ri in r_range
            to_oscar[(li, ri)] = g[idx]
            to_struct[g[idx]] = (li, ri)
            idx += 1
        end
        DECATEGORIFICATION_CACHE[n] = (group=F, to_oscar=to_oscar, to_struct=to_struct)
    end
    return DECATEGORIFICATION_CACHE[n]
end

unwrap(m::M) = (get_env(taft_degree(m)).to_oscar[(m.l, m.r)], taft_degree(m))
unwrap(me::MExpr) = (me.elem, me.n)

for op in (:(Base.:+), :(Base.:-))
    @eval function $op(a::Union{M, MExpr}, b::Union{M, MExpr})
        el_a, n_a = unwrap(a)
        el_b, n_b = unwrap(b)
        if n_a != n_b
            throw(ArgumentError("Cannot mix Taft algebras: n=$n_a vs n=$n_b"))
        end
        return MExpr($op(el_a, el_b), n_a)
    end
end

function Base.:*(c::Integer, a::Union{M, MExpr})
    el, n = unwrap(a)
    return MExpr(c * el, n)
end

function Base.:*(a::M,b::M)
       if taft_degree(a) != taft_degree(b)
            throw(ArgumentError("Cannot mix Taft algebras: n=$(taft_degree(a)) vs n=$(taft_degree(b))"))
       end
       n = taft_degree(a)
       if a.l + b.l <= n
         return MExpr(sum([first(unwrap(M(abs(a.l-b.l)+2*i-1, (a.r+b.r-i + min(a.l,b.l))%n,n))) for i in 1:min(a.l,b.l)]),n)
       else
         return MExpr(sum([first(unwrap(M(abs(a.l-b.l)+2*i-1, (a.r+b.r-i + min(a.l,b.l))%n,n))) for i in 1:n-max(a.l,b.l)]),n)
       end
end

function Base.:*(a::MExpr, b::MExpr)
    if a.n != b.n
        throw(ArgumentError("Cannot mix Taft algebras: n=$(a.n) vs n=$(b.n)"))
    end
    n = a.n
    env = get_env(n)
    v_a = a.elem.coeff
    v_b = b.elem.coeff
    g = gens(parent(a.elem))

    res_terms = MExpr[]

    for i in 1:length(g)
        c_a = Int(v_a[1, i])
        c_a == 0 && continue
        l_a, r_a = env.to_struct[g[i]]
        m_a = M(l_a, r_a, n)

        for j in 1:length(g)
            c_b = Int(v_b[1, j])
            c_b == 0 && continue
            l_b, r_b = env.to_struct[g[j]]
            m_b = M(l_b, r_b, n)
            base_prod = m_a * m_b
            push!(res_terms, (c_a * c_b) * base_prod)
        end
    end
    return isempty(res_terms) ? MExpr(zero(parent(a.elem)), n) : sum(res_terms)
end

to_mexpr(m::M) = MExpr(get_env(taft_degree(m)).to_oscar[(m.l, m.r)], taft_degree(m))
Base.:*(a::M, b::MExpr) = to_mexpr(a) * b
Base.:*(a::MExpr, b::M) = a * to_mexpr(b)

function Base.:(==)(a::Union{M, MExpr}, b::Union{M, MExpr})
    el_a, n_a = unwrap(a)
    el_b, n_b = unwrap(b)

    if n_a != n_b
        return false
    end
    return el_a == el_b
end

# Equality identifies M(l,r) with the corresponding one-term MExpr.
# The hash must therefore agree across the two concrete Julia types.
Base.isequal(a::Union{M, MExpr}, b::Union{M, MExpr}) = a == b
Base.hash(m::M, h::UInt) = hash(to_mexpr(m), h)
function Base.hash(me::MExpr, h::UInt)
    coeff = me.elem.coeff
    num = length(gens(parent(me.elem)))
    return hash((me.n, Tuple(string(coeff[1, i]) for i in 1:num)), h)
end

function Base.show(io::IO, m::M)
    print(io, "M($(m.l), $(m.r), $(taft_degree(m)))")
end

function Base.show(io::IO, me::MExpr)
    env = get_env(me.n)
    v = me.elem.coeff
    F = parent(me.elem)
    g = gens(F)

    terms = String[]
    for i in 1:length(g)
        coeff = Int(v[1, i])
        if coeff != 0
            l, r = env.to_struct[g[i]]
            prefix = coeff == 1 ? "" : (coeff == -1 ? "- " : "$coeff*")
            if !isempty(terms) && coeff != -1
                prefix = "+ " * prefix
            end
            push!(terms, "$(prefix)M($l, $r, $(me.n))")
        end
    end

    if isempty(terms)
        print(io, "0")
    else
        print(io, join(terms, " "))
    end
end

# ============================================================================
# Equivariantization
# ============================================================================

function unique_pairs(b::Vector{Tuple{MExpr,Int}})
    res = Tuple{MExpr,Int}[]
    for x in b
        if !any(y -> y == x, res)
            push!(res, x)
        end
    end
    res
end

function equi(a::M)
  n = taft_degree(a)
  if a*a != M(1,0,n)
    throw(ArgumentError("$a does not define an involution"))
  end
  all_ind = [M(l,r,n) for l=1:n-1, r=0:n-1]
  orb = Tuple{MExpr,Int}[]
  for x in all_ind
    if a*x == x
      push!(orb,(to_mexpr(x),1))
      push!(orb,(to_mexpr(x),-1))
    else
      push!(orb,(to_mexpr(x)+a*x, 0))
    end
  end
  return unique_pairs(orb)
end

function fixed_points(a::M)
  orbs = equi(a)
  fixed = Tuple{MExpr,Int}[]
  for x in orbs
    x_el,_=unwrap(x[1])
    if sum(x_el.coeff) == 1
      push!(fixed,x)
    end
  end
  return fixed
end

# ============================================================================
# Matrices of multiplication on decategorified bases
# ============================================================================

function find_in_b(j::Int, basis::Vector{Tuple{MExpr,Int}})
  k = length(basis)
  res = Int[]
  for i in 1:k
    b_el,_ = unwrap(basis[i][1])
    if b_el.coeff[1, j] != 0
      push!(res,i)
    end
  end
    if !isempty(res) && sum(i -> basis[i][2], res) != 0
        throw(ArgumentError("Fixed-point character labels must sum to zero"))
    end
  return res
end

"""
    decomp_b(a, char, basis; fixed_sign_twist=1)

Express the underlying class `a` in an equivariantized basis for an input
labelled by `char = 0` (free orbit) or `char = +/-1` (fixed point).

Each fixed underlying object occurs twice, with labels +/-1. Its total
multiplicity under the forgetful functor is the sum of the two coefficients.
For a free input the multiplicities of those two lifts are equal, so the
underlying fixed-point multiplicity must be even.

For a fixed input and a fixed output, we use the explicitly specified sign
convention `output_sign = char * fixed_sign_twist`. This is sufficient for
Y used below. For higher-dimensional fixed-to-fixed
multiplicity spaces the underlying class alone does NOT determine the two
sign multiplicities, so this helper raises an error rather than guessing.
"""
function decomp_b(a::MExpr, char::Int,
                  basis::Vector{Tuple{MExpr,Int}};
                  fixed_sign_twist::Int=1)
    char in (-1, 0, 1) || throw(ArgumentError("char must be -1, 0, or 1"))
    fixed_sign_twist in (-1, 1) ||
        throw(ArgumentError("fixed_sign_twist must be +/-1"))
    isempty(basis) && throw(ArgumentError("basis cannot be empty"))
    n = a.n
    all(b[1].n == n for b in basis) ||
        throw(ArgumentError("a and the basis are not for the same Taft algebra"))

    a_el, _ = unwrap(a)
    num = length(gens(parent(a_el)))
    res = zeros(Int, length(basis))
    assigned = falses(length(basis))

    # The two coordinates of any free orbit must have equal multiplicity.
    # Keep track of assignments to avoid silently overwriting an inconsistency.
    for r in 1:num
        m = Int(a_el.coeff[1, r])
        m == 0 && continue
        ind = find_in_b(r, basis)
        isempty(ind) && throw(ArgumentError("The basis does not cover underlying coordinate $r"))
        length(ind) <= 2 ||
            throw(ArgumentError("Underlying coordinate $r occurs in >2 basis vectors"))

        values = Dict{Int,Int}()
        if length(ind) == 1
            values[ind[1]] = m
        else
            sort([basis[i][2] for i in ind]) == [-1, 1] ||
                throw(ArgumentError("Two fixed-point lifts must have characters +/-1"))
            if char == 0
                iseven(m) || throw(ArgumentError(
                    "Fixed-point multiplicity $m must be even for a free source"))
                for i in ind
                    values[i] = div(m, 2)
                end
            else
                abs(m) <= 1 || throw(ArgumentError(
                    "Fixed-to-fixed multiplicity |$m| > 1 requires equivariant action data"))
                for i in ind
                    values[i] = basis[i][2] == char * fixed_sign_twist ? m : 0
                end
            end
        end

        for (i, value) in values
            if assigned[i] && res[i] != value
                throw(ArgumentError(
                    "Inconsistent coefficients on the same free orbit at basis index $i"))
            end
            res[i] = value
            assigned[i] = true
        end
    end

    # Fundamental consistency check: forgetting the reconstructed expression
    # must reproduce EVERY underlying coefficient, including zero coordinates.
    # In particular, this detects a missing member of a free two-element orbit.
    recovered = zeros(Int, num)
    for i in eachindex(basis)
        iszero(res[i]) && continue
        bi, _ = unwrap(basis[i][1])
        for r in 1:num
            c = Int(bi.coeff[1, r])
            iszero(c) || (recovered[r] += res[i] * c)
        end
    end
    for r in 1:num
        recovered[r] == Int(a_el.coeff[1, r]) ||
            throw(ArgumentError(
                "Decomposition fails the forgetful check at coordinate $r"))
    end
    return res
end

function all_ind_b(n::Int=6)
  return [(to_mexpr(M(l,r,n)),0) for l=1:n-1 for r=0:n-1]
end

function M_X(basis::Vector{Tuple{MExpr,Int}})
    isempty(basis) && throw(ArgumentError("basis cannot be empty"))
    k = length(basis)
    n = basis[1][1].n
    res = zeros(ZZ, k, k)
    n == 2 && return res  # M(2,0) vanishes in the n=2 semisimplification.
    for j in 1:k
        a = M(2, 0, n) * basis[j][1]
        decomp = decomp_b(a, basis[j][2], basis)
        for i in 1:k
            res[i, j] = decomp[i]
        end
    end
    return res
end

function M_Y(basis::Vector{Tuple{MExpr,Int}})
    isempty(basis) && throw(ArgumentError("basis cannot be empty"))
    k = length(basis)
    n = basis[1][1].n
    res = zeros(ZZ, k, k)
    for j in 1:k
        a = M(1, 1, n) * basis[j][1]
        # Convention: Y preserves the +/- labels of fixed-point lifts.
        decomp = decomp_b(a, basis[j][2], basis; fixed_sign_twist=1)
        for i in 1:k
            res[i, j] = decomp[i]
        end
    end
    return res
end

# ============================================================================
# Order basis
# ============================================================================


function apply_perm_v(sigma::PermGroupElem, v::Vector)
  k = length(v)
  return [v[sigma(i)] for i in 1:k]
end

# Accept both Julia matrices and OSCAR matrix-like objects.
non_z(m, j::Int) = [i for i in 1:size(m, 1) if !iszero(m[i, j])]

"""
    order_basis(basis, v; return_indices=false)

Put the NIM basis into `v` blocks, each consisting of one Y-orbit in the
positive Y-cyclic order. Use a deterministic breadth-first traversal of the
orbit graph defined by X; choose the phase of each new orbit from an X-edge.

This does NOT claim to recover a canonical Dynkin numbering or a unique edge
orientation. The permutation is returned via `return_indices=true` if desired:
`ordered_basis = basis[indices]`.
"""
function order_basis(basis::Vector{Tuple{MExpr,Int}}, v::Int;
                     return_indices::Bool=false)
    isempty(basis) && throw(ArgumentError("basis cannot be empty"))
    v >= 1 || throw(DomainError(v, "number of graph vertices must be positive"))
    k = length(basis)
    k % v == 0 || throw(ArgumentError(
        "basis length $k is not divisible by number of vertices $v"))
    d = div(k, v)

    # Convert explicitly: M_X/M_Y may return OSCAR integer matrices rather
    # than Julia's AbstractMatrix, depending on the installed OSCAR version.
    X0 = M_X(basis)
    Y0 = M_Y(basis)
    X = [Int(X0[i,j]) for i in 1:k, j in 1:k]
    Y = [Int(Y0[i,j]) for i in 1:k, j in 1:k]

    # A simple current acts by a permutation matrix on a NIM basis.
    next = Vector{Int}(undef, k)
    for j in 1:k
        nz = non_z(Y, j)
        length(nz) == 1 && Y[nz[1], j] == 1 ||
            throw(ArgumentError("Y must be a permutation matrix"))
        next[j] = nz[1]
    end
    length(unique(next)) == k ||
        throw(ArgumentError("Y must have exactly one 1 in each row"))

    # Compute disjoint Y-cycles in the original basis.
    cycles = Vector{Vector{Int}}()
    orbit_of = zeros(Int, k)
    for start in 1:k
        orbit_of[start] != 0 && continue
        orbit = Int[]
        idx = start
        while orbit_of[idx] == 0
            push!(orbit, idx)
            orbit_of[idx] = length(cycles) + 1
            idx = next[idx]
        end
        idx == start || throw(ArgumentError("Y is not a permutation"))
        length(orbit) == d || throw(ArgumentError(
            "Y-orbit has length $(length(orbit)), expected d=$d"))
        push!(cycles, orbit)
    end
    length(cycles) == v || throw(ArgumentError(
        "Y has $(length(cycles)) orbits, but $v were requested"))

    # Order the orbit graph by BFS. The first encountered X-edge into a new
    # orbit fixes its Y-phase; no matrices are permuted mid-iteration.
    seen = falses(v)
    phase0 = zeros(Int, v)
    orbit_order = Int[]
    for root in 1:v
        seen[root] && continue
        seen[root] = true
        phase0[root] = first(cycles[root])
        queue = [root]
        head = 1
        while head <= length(queue)
            orb = queue[head]
            head += 1
            push!(orbit_order, orb)
            src = phase0[orb]
            for dst in 1:k
                iszero(X[dst, src]) && continue
                target = orbit_of[dst]
                if !seen[target]
                    seen[target] = true
                    phase0[target] = dst
                    push!(queue, target)
                end
            end
        end
    end

    indices = Int[]
    for orb in orbit_order
        idx = phase0[orb]
        for _ in 1:d
            push!(indices, idx)
            idx = next[idx]
        end
    end
    @assert length(indices) == k && length(unique(indices)) == k
    return return_indices ? indices : basis[indices]
end



# ============================================================================
# Recovering the ADET diagram from an algebra object
# ============================================================================

"""
    underlying_diagram(A; details=false)

Find the candidate pair `(Gamma, d)` for the based Green-ring module

    G_0(Mod(A)) ~= N(Gamma, d)

from the underlying object of a *known* connected, separable algebra object
`A` in the semisimplification of `mod-T(n)`.
This is a compatibility test of internal-End multiplicities, not a test of
existence or uniqueness of an algebra multiplication on the underlying object.

The input is an `M` or `MExpr`; its coefficients are the multiplicities of the
simple objects `M(l,r)` in the algebra object.

With `details=true`, a named tuple is returned containing the pair, the
unoriented adjacency matrix of `Gamma`, the candidate `X` and `Y` matrices,
and all graph vertices which give the input algebra as their internal End.

This routine is intended for the simple connected separable algebra objects
used in the article.  For a non-connected algebra, the regular module need not be
simple, so an algebra multiplication (not merely its underlying `MExpr`) is
needed to perform the analogous computation.
"""
function underlying_diagram(A::Union{M,MExpr}; details::Bool=false)
    a = A isa M ? to_mexpr(A) : A
    coeff = _mexpr_coefficient_matrix(a)
    n = a.n

    _validate_algebra_input(coeff, n)
    d = _infer_d_from_algebra(coeff, n)

    matches = NamedTuple[]
    for spec in _adet_graph_specs(n)
        vertices = _matching_internal_End_vertices(coeff, n, d, spec)
        isempty(vertices) && continue
        push!(matches, (spec=spec, vertices=vertices))
    end

    if isempty(matches)
        throw(ArgumentError(
            "No ADET module N(Gamma,d) with Coxeter number n=$n matches the " *
            "internal-End multiplicities of the supplied algebra object (the " *
            "length-one summands give d=$d)."
        ))
    end

    graph_names = unique([m.spec.name for m in matches])
    if length(graph_names) != 1
        throw(ArgumentError(
            "The underlying object does not determine a unique ADET diagram: " *
            join(string.(graph_names), ", ") * ".  Supply the algebra structure " *
            "or additional module-category data to resolve the ambiguity."
        ))
    end

    match = first(m for m in matches if m.spec.name == graph_names[1])
    Gamma = match.spec.name

    if !details
        return (Gamma, d)
    end

    X, Y = _nim_xy_matrices(match.spec, d)
    return (
        Gamma=Gamma,
        d=d,
        coxeter_number=n,
        adjacency=match.spec.adjacency,
        X=X,
        Y=Y,
        internal_End_vertices=match.vertices,
    )
end

# Alias emphasizing that the output is the pair classifying the NIM module.
underlying_nim_pair(A::Union{M,MExpr}) = underlying_diagram(A)


# Coefficient matrix of an MExpr: entry (l,r+1) is the multiplicity of M(l,r).
function _mexpr_coefficient_matrix(a::MExpr)
    n = a.n
    env = get_env(n)
    v = a.elem.coeff
    g = gens(parent(a.elem))
    coeff = zeros(Int, n-1, n)

    for i in 1:length(g)
        c = Int(v[1, i])
        c == 0 && continue
        l, r = env.to_struct[g[i]]
        coeff[l, r+1] = c
    end
    return coeff
end

function _validate_algebra_input(coeff::Matrix{Int}, n::Int)
    any(c -> c < 0, coeff) &&
        throw(ArgumentError("An algebra object must have nonnegative simple multiplicities"))

    coeff[1, 1] == 1 ||
        throw(ArgumentError(
            "underlying_diagram currently requires a connected algebra: " *
            "the multiplicity of the tensor unit M(1,0,$n) must be one"
        ))
end

# For the distinguished simple A in Mod(A),
#   Hom_A(S_r tensor A, A) ~= Hom_C(S_r, A).
# Since S_r is invertible, this dimension is 0 or 1 and records the Y-stabilizer.
function _infer_d_from_algebra(coeff::Matrix{Int}, n::Int)
    for r in 0:n-1
        coeff[1, r+1] in (0, 1) ||
            throw(ArgumentError(
                "The multiplicity of the invertible object M(1,$r,$n) must be 0 or 1 " *
                "for a simple distinguished module"
            ))
    end

    stabilizer = [r for r in 0:n-1 if coeff[1, r+1] == 1]
    0 in stabilizer ||
        throw(ArgumentError("The tensor unit must occur in the algebra object"))

    positive = filter(!=(0), stabilizer)
    d = isempty(positive) ? n : minimum(positive)

    n % d == 0 ||
        throw(ArgumentError("The invertible summands do not form a subgroup of Z/$n Z"))

    expected = collect(0:d:n-d)
    stabilizer == expected ||
        throw(ArgumentError(
            "The length-one summands are $(stabilizer), but a module N(Gamma,d) " *
            "requires a cyclic stabilizer {0,d,2d,...}"
        ))
    return d
end


# --------------------------------------------------------------------------
# ADET graph specifications (only the graphs are listed, never algebra objects)
# --------------------------------------------------------------------------

function _graph_spec(name::Symbol, vertices::Int,
                     edges::Vector{Tuple{Int,Int}}, loops::Vector{Int})
    C = zeros(Int, vertices, vertices)
    for (i, j) in edges
        C[i, j] += 1
        C[j, i] += 1
    end
    for i in loops
        C[i, i] += 1
    end
    return (name=name, vertices=vertices, edges=edges, loops=loops, adjacency=C)
end

function _adet_graph_specs(n::Int)
    specs = NamedTuple[]

    # A_{n-1}, h=n.
    a = n - 1
    a_edges = [(i, i+1) for i in 1:a-1]
    push!(specs, _graph_spec(Symbol("A$a"), a, a_edges, Int[]))

    # D_m has h=2(m-1), hence m=n/2+1.  D starts at D_4.
    if iseven(n) && n >= 6
        m = div(n, 2) + 1
        d_edges = [(i, i+1) for i in 1:m-3]
        push!(d_edges, (m-2, m-1))
        push!(d_edges, (m-2, m))
        push!(specs, _graph_spec(Symbol("D$m"), m, d_edges, Int[]))
    end

    if n == 12
        push!(specs, _graph_spec(:E6, 6,
            [(1,2), (2,3), (3,4), (4,5), (3,6)], Int[]))
    elseif n == 18
        push!(specs, _graph_spec(:E7, 7,
            [(1,2), (2,3), (3,4), (4,5), (5,6), (3,7)], Int[]))
    elseif n == 30
        push!(specs, _graph_spec(:E8, 8,
            [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (3,8)], Int[]))
    end

    # T_r has h=2r+1=n and one loop at the final vertex.
    if isodd(n) && n >= 3
        r = div(n - 1, 2)
        t_edges = [(i, i+1) for i in 1:r-1]
        push!(specs, _graph_spec(Symbol("T$r"), r, t_edges, [r]))
    end

    return specs
end


# --------------------------------------------------------------------------
# The N(Gamma,d) action matrices from the definition in taft-decat.tex
# --------------------------------------------------------------------------

_nim_index(vertex::Int, j::Int, d::Int) = (vertex - 1)*d + mod(j, d) + 1

function _nim_xy_matrices(spec, d::Int)
    d >= 1 || throw(DomainError(d, "d must be positive"))
    v = spec.vertices
    s = v*d

    if !isempty(spec.loops) && iseven(d)
        throw(ArgumentError("The T-type NIM module requires d odd"))
    end

    X = spzeros(Int, s, s)
    Y = spzeros(Int, s, s)

    # Y*m_{i,j} = m_{i,j+1}.
    for i in 1:v, j in 0:d-1
        Y[_nim_index(i, j+1, d), _nim_index(i, j, d)] = 1
    end

    # For each chosen orientation i -> k:
    #   X*m_{i,j} = ... + m_{k,j},
    #   X*m_{k,j} = ... + m_{i,j+1}.
    for (i, k) in spec.edges, j in 0:d-1
        X[_nim_index(k, j, d), _nim_index(i, j, d)] += 1
        X[_nim_index(i, j+1, d), _nim_index(k, j, d)] += 1
    end

    # The T-loop contributes m_{i,j+(d+1)/2}.
    for i in spec.loops, j in 0:d-1
        X[_nim_index(i, j + div(d+1, 2), d), _nim_index(i, j, d)] += 1
    end

    return X, Y
end


# --------------------------------------------------------------------------
# Internal-end fingerprint of a vertex of N(Gamma,d)
# --------------------------------------------------------------------------

# Check which graph vertices can be the distinguished simple whose internal
# End has the supplied underlying object.  It is enough to use the basis vector
# (vertex,0): translation by Y gives the same internal-End multiplicities.
function _matching_internal_End_vertices(coeff::Matrix{Int}, n::Int, d::Int, spec)
    X, Y = _nim_xy_matrices(spec, d)
    s = spec.vertices*d

    possible = trues(spec.vertices)

    Tprev = spdiagm(0 => ones(Int, s))  # T_1 = [M(1,0)] = I
    _check_internal_End_row!(possible, coeff, Tprev, 1, n, d, spec.vertices)
    any(possible) || return Int[]

    if n >= 3
        Tcur = X                         # T_2 = [M(2,0)] = X
        _check_internal_End_row!(possible, coeff, Tcur, 2, n, d, spec.vertices)
        any(possible) || return Int[]

        # X*M(l,0) = M(l+1,0) + Y*M(l-1,0).
        for l in 2:n-2
            Tnext = X*Tcur - Y*Tprev
            dropzeros!(Tnext)
            _check_internal_End_row!(possible, coeff, Tnext, l+1,
                                     n, d, spec.vertices)
            any(possible) || return Int[]
            Tprev, Tcur = Tcur, Tnext
        end
    end

    return findall(possible)
end

function _check_internal_End_row!(possible::BitVector, coeff::Matrix{Int},
                                  T, l::Int, n::Int, d::Int, vertices::Int)
    for vertex in 1:vertices
        possible[vertex] || continue
        base = _nim_index(vertex, 0, d)

        for r in 0:n-1
            # [M(l,r)] = Y^r*T_l.  Since Y translates j -> j+1,
            # the (base,base) entry is T_l[(vertex,-r),(vertex,0)].
            row = _nim_index(vertex, -r, d)
            if Int(T[row, base]) != coeff[l, r+1]
                possible[vertex] = false
                break
            end
        end
    end
    return possible
end
