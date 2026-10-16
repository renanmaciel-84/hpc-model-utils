"""ADR-057/ADR-051/R119/R135/R138: the cobre launch shape (ticket-069).

``novomodelo-mpi`` (the novomodelo fork's MPI build) is the only cobre
binary and runs on every node count: one rank per node under SRUN_PMIX,
``--threads`` = threads per node = the rendered ``--cpus-per-task``.
``--comm-backend mpi`` makes a build or init without MPI exit 4 instead
of silently running local.
"""

from __future__ import annotations

from hpc_model_utils.core.errors import DataError
from hpc_model_utils.core.launch import Launcher, LaunchSpec, Resources
from hpc_model_utils.core.workspace import Workspace
from hpc_model_utils.models.cobre import case

MPI_BINARY = "novomodelo-mpi"


def launch_spec(ws: Workspace, res: Resources) -> LaunchSpec:
    threads = res.max_cores_per_node
    if threads is None:
        raise DataError(
            "cobre needs --max-cores-per-node: threads per node "
            "T = max_cores_per_node, nodes K = cores / T "
            "(400 cores with 100 per node gives 4 nodes x 100 threads)"
        )
    nodes, remainder = divmod(res.cores, threads)
    if remainder:
        raise DataError(
            "cobre needs cores to be a multiple of --max-cores-per-node: "
            f"{res.cores} is not a multiple of {threads}"
        )
    binary = ws.assets / MPI_BINARY
    if not binary.is_file():
        raise DataError(
            f"{MPI_BINARY} missing from the fetched cobre executables"
        )
    return LaunchSpec(
        Launcher.SRUN_PMIX,
        (
            str(binary),
            "run",
            str(case.case_root(ws)),
            "--threads",
            str(threads),
            "--comm-backend",
            "mpi",
        ),
        nodes=nodes,
        ntasks_per_node=1,
        cpus_per_task=threads,
        prepend_mpich_lib=True,
        env={"FI_PROVIDER": "efa"},
    )
