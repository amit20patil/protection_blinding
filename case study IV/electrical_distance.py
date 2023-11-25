'''Methods to get the electrical equivalence impedance between two buses'''
import logging
import warnings
from itertools import product

import numpy as np
import pandapower as pp
from pandapower.grid_equivalents import build_ppc_and_Ybus
from scipy import sparse
from scipy.linalg import pinv
from scipy.sparse import spmatrix
from scipy.sparse.linalg import inv as inv_sparse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_equivalent_impedance(
    net: pp.pandapowerNet,
    i: int,
    j: int,
    ohm: bool = True,
    recalculate_ppc: bool = True,
) -> complex:
    """Method to calculate the equivalent impedance (=electrical distance) between two buses in
    a pandapower network

    Method adapted from [1,2]
    [1] https://aip.scitation.org/doi/pdf/10.1063/1.3077229
    [2] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6241462

    Ze_ij = z_ii - 2 * z_ij + z_jj

    Args:
        net (pp.pandapowerNet): network
        i (int): index of first bus
        j (int): index of second bus
        ohm (bool, optional): if True the result will be in ohm, otherwise in p.u. of base voltage.
            Defaults to True.
        recalculate_ppc (bool, optional): Flag to recalculate ppc and thus YBus matrix.
            Defaults to True

    Returns:
        complex: complex impedance
    """
    # Get YBus matrix
    if (
        recalculate_ppc
        or net["_ppc"] is None
        or net["_ppc"]["internal"]["Ybus"].shape[0] < 1
    ):
        log.debug("Building PPC and YBus.")
        build_ppc_and_Ybus(net)
    y_bus_matrix = net["_ppc"]["internal"]["Ybus"]

    # Convert from p.u. to ohm if enabled
    if ohm:
        base_kv = np.zeros(shape=y_bus_matrix.shape)
        for a, b in list(
            product(range(y_bus_matrix.shape[0]), range(y_bus_matrix.shape[1]))
        ):
            base_a = pp.topology.get_baseR(net, net["_ppc"], a)
            base_b = pp.topology.get_baseR(net, net["_ppc"], b)
            base_kv[a][b] = max(base_a, base_b)
        y_bus_matrix = sparse.csr_matrix(y_bus_matrix.toarray() / base_kv)

    # Invert YBus matrix to get ZBus matrix
    z_bus_matrix = _invert_matrix(y_bus_matrix)

    # Calculate equivalent impedance
    z_ii = z_bus_matrix.item(i, i)
    z_ij = z_bus_matrix.item(i, j)
    z_jj = z_bus_matrix.item(j, j)

    return z_ii - 2 * z_ij + z_jj


def _invert_matrix(matrix: spmatrix) -> spmatrix:
    """Function to invert a given matrix

    Args:
        matrix (spmatrix): given matrix

    Raises:
        exc: if inversion is not possible

    Returns:
        spmatrix: inverted matrix
    """
    try:
        sparsity = matrix.nnz / matrix.shape[0] ** 2
        if sparsity < 0.002:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return inv_sparse(matrix).toarray()
        else:
            # Using pseudo inverse (slower but better results for small impedance lines)
            return pinv(matrix.toarray())
    except ValueError as exc:
        raise exc
