/* LINEAL library test, by Kurt Böhm */


#include <cstdint>
#include <iostream>
#include <memory>

#include "linealia.h"
#include "linealiaLib.h"

using u8 = std::uint8_t;
using u32 = std::uint32_t;

int main()
{
    const u32 size = 73;

    if (! LinealiaLib::instance().load())
        return -1;

    // LHS:
    //  2.25 -1     0     0     …     0
    // -1     2.25 -1     0     …     0
    //  0    -1     2.25 -1     …     0
    //  ⋮     ⋱     ⋱     ⋱     ⋱     ⋮
    //  0     …     0    -1     2.25 -1
    //  0     …     0     0    -1     2.25
    // RHS[i] = 1 if (i == 0 or i + 1 == size) else 0

    // Names in double quotes are in reference to the data description which you sent me a while ago

    // A flat array that stores the values which "values" references
    // This is just a helper for "MatrixCPU::values" to avoid many tiny allocations
    auto lhs_fentries = std::make_unique_for_overwrite<double[]>(3 * size);
    // "MatrixCPU::values" with each row taking up 2-3 entries
    auto lhs_entries = std::make_unique_for_overwrite<double*[]>(size);
    // A flat array that stores the columns which "MatrixCPU::colIndeces" references
    auto lhs_fcolumns = std::make_unique_for_overwrite<u32[]>(3 * size);
    // "MatrixCPU::colIndeces" with each row taking up 2-3 entries
    auto lhs_columns = std::make_unique_for_overwrite<u32*[]>(size);
    // "MatrixCPU::numColumns", which is 3 for all rows apart from the first and last, where it is 2
    auto lhs_cnums = std::make_unique_for_overwrite<u8[]>(size);

    // "VectorCPU::values"
    auto rhs_values = std::make_unique_for_overwrite<double[]>(size);

    for (u32 i = 0; i < size; ++i)
    {
        const u32 old_offset = i*3;
        u32 offset = old_offset;

        // Set "MatrixCPU::values" and "MatrixCPU::colIndeces" so that they point to the correct
        // part of the flat arrays
        lhs_entries[i] = lhs_fentries.get() + old_offset;
        lhs_columns[i] = lhs_fcolumns.get() + old_offset;

        // The order in which the entries of each row are added is deliberately not ascending
        // to ensure the implementation does not depend on ordered entries in each row

        // main diagonal
        double value = 0;
        {
            const double v = 2.25;
            value += v;
            lhs_fentries[offset] = v;
            lhs_fcolumns[offset] = i;
            ++offset;
        }

        // left of main diagonal
        if (i > 0)
        {
            const double v = -1;
            value += v;
            lhs_fentries[offset] = v;
            lhs_fcolumns[offset] = i - 1;
            ++offset;
        }

        // right of main diagonal
        if (i + 1 < size)
        {
            const double v = -1;
            value += v;
            lhs_fentries[offset] = v;
            lhs_fcolumns[offset] = i + 1;
            ++offset;
        }

        // Store the number of entries in row i and the RHS value at i
        lhs_cnums[i] = u8(offset - old_offset);
        rhs_values[i] = value;
    }

    std::cout << "lhs:\n";
    for (u32 i = 0; i < size; ++i)
    {
        std::cout << i << ": ";
        const auto cnum = lhs_cnums[i];
        for (u32 j = 0; j < cnum; ++j)
        {
            if (j > 0)
                std::cout << ", ";

            lhs_entries[i][j] /= 2.25;
            std::cout << lhs_columns[i][j] << ":" << lhs_entries[i][j];
        }
        std::cout << "\n";
    }

    std::cout << "rhs: ";
    for (u32 i = 0; i < size; ++i)
    {
        if (i > 0)
            std::cout << ", ";

        rhs_values[i] /= 2.25;
        std::cout << rhs_values[i];
    }
    std::cout << "\n";

    LinealiaMatrix lhs{};
    lhs.num_rows = size;
    lhs.max_columns = 3;
    lhs.num_columns = lhs_cnums.get();
    lhs.column_indices = lhs_columns.get();
    lhs.values = lhs_entries.get();

    LinealiaVector rhs{};
    rhs.num_elements = size;
    rhs.values = rhs_values.get();

    LinealiaIterativeSolverParams iter_params{};
    iter_params.max_iterations = 128;
    iter_params.max_relative_residual_norm = 1e-10;

    auto sol_values = std::make_unique<double[]>(size);
    LinealiaVector sol{};
    sol.num_elements = size;
    sol.values = sol_values.get();

    {
        std::cout << "\nSOR:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaLib::instance().solveSOR(lhs, sol, rhs, {}, iter_params, {});
    }

    {
        std::cout << "\nSSOR:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaLib::instance().solveSSOR(lhs, sol, rhs, {}, iter_params, {});
    }

    {
        std::cout << "\nCG:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaLib::instance().solveCG(lhs, sol, rhs, {}, iter_params);
    }

    {
        std::cout << "\nPCG_SOR:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaRelaxedPreconditionerParams relax_params{};
        relax_params.iterations = 5;
        LinealiaLib::instance().solvePCG_SOR(lhs, sol, rhs, {}, iter_params, relax_params);
    }

    {
        std::cout << "\nPCG_SSOR:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaRelaxedPreconditionerParams relax_params{};
        relax_params.iterations = 1;
        LinealiaLib::instance().solvePCG_SSOR(lhs, sol, rhs, {}, iter_params, relax_params);
    }

    {
        std::cout << "\nPCG_AMG_SOR:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaLib::instance().solvePCG_AMG_SOR(lhs, sol, rhs, {}, iter_params, {});
    }

    {
        std::cout << "\nPCG_AMG_SSOR:\n";

        for (u32 i = 0; i < size; ++i)
        { sol_values[i] = 0.5; }

        LinealiaLib::instance().solvePCG_AMG_SOR(lhs, sol, rhs, {}, iter_params, {});
    }

    std::cout << "X: ";
    for (u32 i = 0; i < size; ++i)
    {
        if (i > 0)
            std::cout << ", ";

        std::cout << sol_values[i];
    }
    std::cout << "\n";
}
