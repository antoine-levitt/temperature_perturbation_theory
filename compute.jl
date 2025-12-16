using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData
using JLD2
using FFTW

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)
DFTK.disable_threading()

a = 7.6324708938577865

lattice = [0 a/2 a/2;
           a/2 0 a/2; 
           a/2 a/2 0]
Al = ElementPsp(:Al, PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
atoms = [Al]
positions = [[0, 0, 0]]
nk = 40

function get(T)
    model = model_DFT(lattice, atoms, positions; functionals=LDA(), temperature=T)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[nk, nk, nk])
    scfres = self_consistent_field(basis, tol=1e-12, mixing=KerkerMixing())
    [scfres.energies.total, scfres.energies.Entropy / T]
end
# T0 = .01
# get(T0)
# stop
# derivative_ε = let ε = 1e-5
#     (get(T0+ε) - get(T0-ε)) / 2ε
# end
# derivative_fd = ForwardDiff.derivative(get, T0)
# @test norm(derivative_ε - derivative_fd) < 1e-4

Ts = collect(range(0.001,.01,step=.001))
# Ts = collect(range(0.01,.1,step=.02))
Es = zeros(length(Ts))
Ss = zeros(length(Ts))
Ds = zeros(length(Ts))

for (iT, T) in enumerate(Ts)
    Es[iT], _ = get(T)
    Ss[iT], Ds[iT] = ForwardDiff.derivative(get, T)
end
@save "$nk.jld2" Ts Es Ss Ds nk
