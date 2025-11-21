// This file is part of https://github.com/KurtBoehm/linealia.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEALIA_LINEALIA_HPP
#define INCLUDE_LINEALIA_LINEALIA_HPP

#include <cstddef>
#include <cstdint>

extern "C" {
enum LinealiaIterativeStopReason {
  LINEALIA_STOP_RESIDUAL_NORM,
  LINEALIA_STOP_ITERATIONS,
};
struct LinealiaIterativeResult {
  LinealiaIterativeStopReason reason;
  size_t iteration;
  // ||Ax−b||₂
  double residual_norm;
  // ||Ax−b||₂/||b||₂
  double relative_residual_norm;
};

struct LinealiaMatrix {
  uint32_t num_rows = 0;
  uint8_t max_columns = 11;
  uint8_t* num_columns = nullptr;
  uint32_t** column_indices = nullptr;
  double** values = nullptr;
};

struct LinealiaVector {
  uint32_t num_elements = 0;
  double* values = nullptr;
};

struct LinealExecutionParams {
  // The number of threads to use
  // If the value is 0 (the default), the number of threads is determined automatically
  size_t num_threads = 0;
};

struct LinealiaIterativeSolverParams {
  // Maximum number of iterations
  size_t max_iterations = 256;
  // Maximum relative residual norm, i.e. ||Ax−b||₂/||b||₂
  double max_relative_residual_norm = 1e-10;
};

struct LinealiaRelaxedParams {
  // The relaxation factor, commonly called ω (1 by default, i.e. SOR→Gauß-Seidel)
  double relax = 1.0;
};
LinealiaIterativeResult linealia_solve_sor(LinealiaMatrix lhs, LinealiaVector sol,
                                           LinealiaVector rhs, LinealExecutionParams eparams,
                                           LinealiaIterativeSolverParams iparams,
                                           LinealiaRelaxedParams sparams);
LinealiaIterativeResult linealia_solve_ssor(LinealiaMatrix lhs, LinealiaVector sol,
                                            LinealiaVector rhs, LinealExecutionParams eparams,
                                            LinealiaIterativeSolverParams iparams,
                                            LinealiaRelaxedParams sparams);

LinealiaIterativeResult linealia_solve_cg(LinealiaMatrix lhs, LinealiaVector sol,
                                          LinealiaVector rhs, LinealExecutionParams eparams,
                                          LinealiaIterativeSolverParams iparams);

struct LinealiaRelaxedPreconditionerParams {
  // The relaxation factor, commonly called ω (1 by default, i.e. Gauß-Seidel)
  double relax = 1.0;
  // The number of SOR iterations to apply per CG iteration
  size_t iterations = 2;
};
LinealiaIterativeResult
linealia_solve_pcg_sor(LinealiaMatrix lhs, LinealiaVector sol, LinealiaVector rhs,
                       LinealExecutionParams eparams, LinealiaIterativeSolverParams iparams,
                       LinealiaRelaxedPreconditionerParams preconditioner_params);
LinealiaIterativeResult
linealia_solve_pcg_ssor(LinealiaMatrix lhs, LinealiaVector sol, LinealiaVector rhs,
                        LinealExecutionParams eparams, LinealiaIterativeSolverParams iparams,
                        LinealiaRelaxedPreconditionerParams preconditioner_params);

struct AmgAggregateSizeRange {
  size_t min;
  size_t max;
};
struct LinealiaPcgAmgParams {
  // The AMG relaxation factor
  double amg_relax = 0.6;
  // The aggregate size ranges
  // If these are null/0, uses [8, 12] on the finest level and [6, 9] on other levels
  AmgAggregateSizeRange* amg_aggregate_size_ranges = nullptr;
  size_t amg_aggregate_size_ranges_size = 0;
  // The maximum number of levels in the AMG hierarchy
  size_t amg_max_level_num = 64;
  // A linear system must have at least this size to continue coarsening
  uint32_t amg_min_coarsen_size = 1024;
  // The decrease in the size of the linear system must be at least this factor
  // to continue coarsening
  double amg_min_coarsen_factor = 1.2;
  // The minimum number of unknowns per thread
  // If there are fewer, the number of threads is reduced accordingly
  double amg_aggregation_min_per_thread = 32768;
  // The SOR smoother relaxation factor, commonly called ω (1 by default)
  double smoother_relax = 1.0;
  // The number of SOR smoother iterations to apply
  size_t smoother_iterations = 2;
};
LinealiaIterativeResult linealia_solve_pcg_amg_sor(LinealiaMatrix lhs, LinealiaVector sol,
                                                   LinealiaVector rhs,
                                                   LinealExecutionParams eparams,
                                                   LinealiaIterativeSolverParams iparams,
                                                   LinealiaPcgAmgParams sparams);
LinealiaIterativeResult linealia_solve_pcg_amg_ssor(LinealiaMatrix lhs, LinealiaVector sol,
                                                    LinealiaVector rhs,
                                                    LinealExecutionParams eparams,
                                                    LinealiaIterativeSolverParams iparams,
                                                    LinealiaPcgAmgParams sparams);
}

#endif // INCLUDE_LINEALIA_LINEALIA_HPP
