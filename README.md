# CLASS with Early Dark Energy parametrization

This repository is a modified version of the public Boltzmann code **CLASS**
(Cosmic Linear Anisotropy Solving System), extended to implement a unified,
parametrized **Early Dark Energy (EDE)** fluid and a **Chevallier–Polarski–Linder
(CPL)** late-time dark energy equation of state. The modifications were
developed to compute the background and perturbation evolution of these
extended dark energy models for a systematic Bayesian comparison against
$\Lambda$CDM, combining DESI DR2 BAO, Pantheon+ Type Ia supernovae, and
*Planck* 2018 CMB data.

Full details of the model, methodology, and results are presented in:

**"Early against Late: A contrast on dark energy in the light of DESI DR2"**
Miguel A. Zapata, Karim Carrion, and Gabriela Garcia-Arroyo
[arXiv:2609.05410](https://arxiv.org/pdf/2609.05410)

**Authors of this code:**
- [Miguel A. Zapata](https://arxiv.org/search/astro-ph?searchtype=author&query=Zapata,+M+A)
- [Karim Carrion](https://arxiv.org/search/astro-ph?searchtype=author&query=Carrion,+K)
- [Gabriela Garcia-Arroyo](https://arxiv.org/search/astro-ph?searchtype=author&query=Garcia-Arroyo,+G)

If you use this code in a publication, please cite the paper above **in
addition to** the fundamental CLASS papers listed below, following the
standard CLASS citation policy.

---

## About the model

The unified EDE fluid tracks the equation of state of the dominant matter
component of the universe at early times, behaving like radiation during
radiation domination and like matter during matter domination, while
maintaining a non-negligible fractional density before transitioning to
drive late-time cosmic acceleration. It is controlled by two additional
free parameters, `w0_fld` and `Omega_EDE`, on top of the standard
cosmological parameters. The CPL parametrization is implemented as a
late-time counterpart, controlled by `w0_fld` and `wa_fld`, and is used
throughout the paper as a benchmark against which the early-time modification
is contrasted. Both models rely on the Parametrized Post-Friedmann (PPF)
scheme already present in CLASS to handle the phantom-divide crossing
($w_{\rm de}=-1$) consistently at the perturbation level.

---

## Original CLASS documentation

The sections below reproduce the standard CLASS documentation. Compilation,
usage, and the Python wrapper work exactly as in the public CLASS release;
no additional installation steps are required for the EDE and CPL
extensions beyond the modified source files already included in this
repository.

### Compiling CLASS and getting started

(the information below can also be found on the webpage, just below the
download button)

Download the code from the webpage and unpack the archive
(`tar -zxvf class_vx.y.z.tar.gz`), or clone this repository directly. Go to
the class directory (`cd class/` or `class_public/` or `class_vx.y.z/`) and
compile (`make clean; make class`). You can usually speed up compilation
with the `-j` option: `make -j class`. If the first compilation attempt
fails, you may need to open the Makefile and adapt the name of the compiler
(default: `gcc`), the optimization flag (default: `-O4 -ffast-math`), and
the OpenMP flag (default: `-fopenmp`; this flag is optional, you are free to
compile without OpenMP if you don't want parallel execution; note that you
need version 4.2 or higher of `gcc` to compile with `-fopenmp`). Many more
details on CLASS compilation are given on the wiki page:

https://github.com/lesgourg/class_public/wiki/Installation

(in particular, for compiling on Mac >= 10.9 despite the clang
incompatibility with OpenMP).

To check that the code runs, type: ./class explanatory.ini


The `explanatory.ini` file is *the* reference input file, containing and
explaining the use of all possible input parameters. We recommend reading
it, keeping it unchanged (for future reference), and creating your own
shorter input files for your own purposes, containing only the input lines
useful to you. Input files must have a `.ini` extension. We provide an
example input file containing a selection of the most-used parameters,
`default.ini`, that you may use as a starting point.

If you want to play with the precision/speed of the code, you can use one
of the provided precision files (e.g., `cl_permille.pre`) or modify one of
them, and run with two input files, for instance:./class test.ini cl_permille.pre


The `*.pre` files are meant to specify the precision parameters for which
you don't want to keep default values. If you find it more convenient, you
can pass these precision parameter values in your `*.ini` file instead of
an additional `*.pre` file.

The automatically generated documentation is located in:
doc/manual/html/index.html
doc/manual/CLASS_manual.pdf


On top of that, if you wish to modify the code, you will find plenty of
comments directly in the source files.

### Python

To use CLASS from Python, IPython notebooks, or from a Monte Carlo
parameter-extraction code such as MontePython or Cobaya, you need to compile
not only the code, but also its Python wrapper. This can be done by typing
just `make` instead of `make class` (or, to speed things up, `make -j`).
More details on the wrapper and its compilation are found on the wiki page:

https://github.com/lesgourg/class_public/wiki

### Developing the code

If you want to develop the code further, we suggest downloading the
original public version from the GitHub webpage:

https://github.com/lesgourg/class_public

rather than from class-code.net. This way, you will have access to all the
features of git repositories, and can develop your own branch. For related
instructions, check:

https://github.com/lesgourg/class_public/wiki/Public-Contributing

---

## Citing this code

You are free to use this code, provided that in your publications you cite,
at minimum, the fork-specific paper above **and** the following fundamental
CLASS papers, in line with the official CLASS citation policy (feel free to
cite more CLASS papers if relevant to your use case):

- D. Blas, J. Lesgourgues, and T. Tram, *"The Cosmic Linear Anisotropy
  Solving System (CLASS) II: Approximation schemes,"* JCAP **07** (2011) 034,
  [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)
- J. Lesgourgues, *"The Cosmic Linear Anisotropy Solving System (CLASS) I:
  Overview,"* [arXiv:1104.2932](https://arxiv.org/abs/1104.2932)
- J. Lesgourgues, *"The Cosmic Linear Anisotropy Solving System (CLASS) III:
  Comparison with CAMB for $\Lambda$CDM,"*
  [arXiv:1104.2934](https://arxiv.org/abs/1104.2934)
- J. Lesgourgues and T. Tram, *"The Cosmic Linear Anisotropy Solving System
  (CLASS) IV: Efficient implementation of non-cold relics,"*
  [arXiv:1104.2935](https://arxiv.org/abs/1104.2935)

## Support

For questions specific to the EDE and CPL extensions implemented in this
fork, please open an issue on this repository. For general CLASS support,
please open a new issue on the main CLASS webpage:

https://github.com/lesgourg/class_public