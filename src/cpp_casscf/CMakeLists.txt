cmake_minimum_required(VERSION 3.16)
project(casscf_trust_region CXX)

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
)
target_include_directories(casscf_trust_region PUBLIC include)
target_link_libraries(casscf_trust_region PUBLIC Eigen3::Eigen)
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

# Standalone tool, not a ctest: replays real Python-captured solver instances
# from a dump directory (see validation/dump_lih_case.py). Run manually:
#   ./validate_against_python <dump_dir>
add_executable(validate_against_python tools/validate_against_python.cpp)
target_link_libraries(validate_against_python PRIVATE casscf_trust_region)
