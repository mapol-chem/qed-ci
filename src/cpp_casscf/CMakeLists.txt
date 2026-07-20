cmake_minimum_required(VERSION 3.16)
project(casscf_trust_region CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen ships in the p4dev conda env (see README.md for other locations).
# Override with -DCMAKE_PREFIX_PATH=/path/to/env if Eigen lives elsewhere.
if(NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/../p4dev" "$ENV{CONDA_PREFIX}/envs/p4dev")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- Plain-C CASSCF backend (ci_solver.c / orbital.c, one level up in
# qed-ci/src/) --- the existing CI Davidson solver / RDM builder /
# integral-transformation C extension this Python codebase already calls
# via ctypes (see the top-level qed-ci/README.md "Compile the code with
# intel compiler" step, and cpp_casscf/README.md's CiStateAverageSolver/
# IntegralTransformer sections). Compiled directly from these tracked .c
# files here (not linked against a prebuilt cfunctions.so) so the C++ port
# depends on exactly the same source everything else does, not a build
# artifact that can silently drift out of sync with it.
#
# Both files call cblas_*/LAPACKE_dsyev (need MKL) and use
# `#pragma omp parallel for` (need an OpenMP runtime) -- override
# MKLROOT if MKL lives somewhere other than the oneAPI/conda locations
# probed below.
set(CASSCF_C_BACKEND_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "Directory containing ci_solver.c/orbital.c (the plain-C CASSCF backend)")

find_path(MKL_INCLUDE_DIR mkl.h
    HINTS ENV MKLROOT
    PATHS ${CMAKE_PREFIX_PATH} /opt/intel/oneapi/mkl/latest /home/nvu12/intel/oneapi/mkl/2024.1
    PATH_SUFFIXES include)
find_library(MKL_RT_LIBRARY NAMES mkl_rt
    HINTS ENV MKLROOT
    PATHS ${CMAKE_PREFIX_PATH} /opt/intel/oneapi/mkl/latest /home/nvu12/intel/oneapi/mkl/2024.1
    PATH_SUFFIXES lib lib/intel64)

if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIBRARY)
    message(FATAL_ERROR "MKL not found (need mkl.h and libmkl_rt) -- "
        "ci_solver.c/orbital.c call cblas_*/LAPACKE_dsyev. Set MKLROOT or "
        "CMAKE_PREFIX_PATH to an MKL install (e.g. an oneAPI mkl/<ver> "
        "directory, or a conda env that ships libmkl_rt.so).")
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

add_library(casscf_c_backend STATIC
    ${CASSCF_C_BACKEND_DIR}/ci_solver.c
    ${CASSCF_C_BACKEND_DIR}/orbital.c
)
target_include_directories(casscf_c_backend PUBLIC ${CASSCF_C_BACKEND_DIR} ${MKL_INCLUDE_DIR})
target_link_libraries(casscf_c_backend PUBLIC ${MKL_RT_LIBRARY} OpenMP::OpenMP_C)
target_compile_options(casscf_c_backend PRIVATE -fPIC)
# No -Wall -Wextra here: this is existing, unmodified C we're linking
# against (already covered by the top-level qed-ci Python test suite),
# not code this port is writing or reviewing line by line.

add_library(casscf_trust_region
    src/pcg_trust_region_solver.cpp
    src/gltr_trust_region_solver.cpp
    src/lstrs_solver.cpp
    src/lstrs_bisection_core.cpp
    src/bordered_eigensolve.cpp
    src/davidson_augmented_hessian_solver.cpp
    src/davidson_driven_lstrs_solver.cpp
    src/gram_schmidt.cpp
    src/jacobi_davidson_correction.cpp
    src/quasi_newton_policy.cpp
    src/solver_selector.cpp
    src/orbital_rotation.cpp
    src/macroiteration_driver.cpp
    src/minres_solver.cpp
    src/intermediates.cpp
    src/hessian_guess.cpp
    src/internal_optimization.cpp
    src/internal_optimization_step.cpp
    src/orbital_sigma.cpp
    src/microiteration_energy.cpp
    src/bfgs_operator.cpp
    src/microiteration_ci_integrals_transform.cpp
    src/microiteration_optimization_step.cpp
    src/integral_transformer.cpp
    src/ci_setup.cpp
    src/ci_state_average_solver.cpp
)
target_include_directories(casscf_trust_region PUBLIC include)
target_link_libraries(casscf_trust_region PUBLIC Eigen3::Eigen casscf_c_backend)
target_compile_options(casscf_trust_region PRIVATE -Wall -Wextra)

enable_testing()

add_executable(test_pcg_trust_region tests/test_pcg_trust_region.cpp)
target_link_libraries(test_pcg_trust_region PRIVATE casscf_trust_region)
add_test(NAME pcg_trust_region COMMAND test_pcg_trust_region)

add_executable(test_lstrs_solver tests/test_lstrs_solver.cpp)
target_link_libraries(test_lstrs_solver PRIVATE casscf_trust_region)
add_test(NAME lstrs_solver COMMAND test_lstrs_solver)

add_executable(test_davidson_augmented_hessian_solver tests/test_davidson_augmented_hessian_solver.cpp)
target_link_libraries(test_davidson_augmented_hessian_solver PRIVATE casscf_trust_region)
add_test(NAME davidson_augmented_hessian_solver COMMAND test_davidson_augmented_hessian_solver)

add_executable(test_gltr_trust_region_solver tests/test_gltr_trust_region_solver.cpp)
target_link_libraries(test_gltr_trust_region_solver PRIVATE casscf_trust_region)
add_test(NAME gltr_trust_region_solver COMMAND test_gltr_trust_region_solver)

add_executable(test_davidson_driven_lstrs_solver tests/test_davidson_driven_lstrs_solver.cpp)
target_link_libraries(test_davidson_driven_lstrs_solver PRIVATE casscf_trust_region)
add_test(NAME davidson_driven_lstrs_solver COMMAND test_davidson_driven_lstrs_solver)

add_executable(test_davidson_expansion_loop tests/test_davidson_expansion_loop.cpp)
target_link_libraries(test_davidson_expansion_loop PRIVATE casscf_trust_region)
add_test(NAME davidson_expansion_loop COMMAND test_davidson_expansion_loop)

add_executable(test_minres_solver tests/test_minres_solver.cpp)
target_link_libraries(test_minres_solver PRIVATE casscf_trust_region)
add_test(NAME minres_solver COMMAND test_minres_solver)

add_executable(test_orbital_rotation tests/test_orbital_rotation.cpp)
target_link_libraries(test_orbital_rotation PRIVATE casscf_trust_region)
add_test(NAME orbital_rotation COMMAND test_orbital_rotation)

add_executable(test_macroiteration_driver tests/test_macroiteration_driver.cpp)
target_link_libraries(test_macroiteration_driver PRIVATE casscf_trust_region)
add_test(NAME macroiteration_driver COMMAND test_macroiteration_driver)

add_executable(test_internal_optimization tests/test_internal_optimization.cpp)
target_link_libraries(test_internal_optimization PRIVATE casscf_trust_region)
add_test(NAME internal_optimization COMMAND test_internal_optimization)

add_executable(test_internal_optimization_step tests/test_internal_optimization_step.cpp)
target_link_libraries(test_internal_optimization_step PRIVATE casscf_trust_region)
add_test(NAME internal_optimization_step COMMAND test_internal_optimization_step)

add_executable(test_orbital_sigma tests/test_orbital_sigma.cpp)
target_link_libraries(test_orbital_sigma PRIVATE casscf_trust_region)
add_test(NAME orbital_sigma COMMAND test_orbital_sigma)

add_executable(test_microiteration_energy tests/test_microiteration_energy.cpp)
target_link_libraries(test_microiteration_energy PRIVATE casscf_trust_region)
add_test(NAME microiteration_energy COMMAND test_microiteration_energy)

add_executable(test_bfgs_operator tests/test_bfgs_operator.cpp)
target_link_libraries(test_bfgs_operator PRIVATE casscf_trust_region)
add_test(NAME bfgs_operator COMMAND test_bfgs_operator)

add_executable(test_microiteration_ci_integrals_transform tests/test_microiteration_ci_integrals_transform.cpp)
target_link_libraries(test_microiteration_ci_integrals_transform PRIVATE casscf_trust_region)
add_test(NAME microiteration_ci_integrals_transform COMMAND test_microiteration_ci_integrals_transform)

add_executable(test_microiteration_optimization_step tests/test_microiteration_optimization_step.cpp)
target_link_libraries(test_microiteration_optimization_step PRIVATE casscf_trust_region)
add_test(NAME microiteration_optimization_step COMMAND test_microiteration_optimization_step)

add_executable(test_ci_orbital_backend_smoke tests/test_ci_orbital_backend_smoke.cpp)
target_include_directories(test_ci_orbital_backend_smoke PRIVATE include)
target_link_libraries(test_ci_orbital_backend_smoke PRIVATE casscf_c_backend)
add_test(NAME ci_orbital_backend_smoke COMMAND test_ci_orbital_backend_smoke)

add_executable(test_integral_transformer tests/test_integral_transformer.cpp)
target_link_libraries(test_integral_transformer PRIVATE casscf_trust_region)
add_test(NAME integral_transformer COMMAND test_integral_transformer)

add_executable(test_ci_state_average_solver tests/test_ci_state_average_solver.cpp)
target_link_libraries(test_ci_state_average_solver PRIVATE casscf_trust_region)
add_test(NAME ci_state_average_solver COMMAND test_ci_state_average_solver)

# Standalone tool, not a ctest: replays real Python-captured solver instances
# from a dump directory (see validation/dump_lih_case.py). Run manually:
#   ./validate_against_python <dump_dir>
add_executable(validate_against_python tools/validate_against_python.cpp)
target_link_libraries(validate_against_python PRIVATE casscf_trust_region)

# Standalone tool, not a ctest: wires all 4 real MacroiterationDriver
# collaborators together and runs the whole thing end to end against real
# captured LiH chemistry (see validation/dump_lih_case.py). Run manually:
#   ./run_macroiteration_driver <dump_dir>
add_executable(run_macroiteration_driver tools/run_macroiteration_driver.cpp)
target_link_libraries(run_macroiteration_driver PRIVATE casscf_trust_region)
