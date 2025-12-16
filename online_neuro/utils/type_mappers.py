# online_neuro/utils/type_mappers.py
from .constants import ProblemType


def get_problem_class(exp_type: str) -> ProblemType:
    """
    Map a string describing the experiment type to an internal ``ProblemType`` enum.

    The mapping is case-insensitive and supports several aliases:

    - ``"multiobjective"`` or ``"moo"`` → ``ProblemType.MULTIOBJECTIVE``
    - ``"classification"`` → ``ProblemType.CLASSIFICATION``
    - Anything else defaults to ``ProblemType.REGRESSION``

    Parameters
    ----------
    exp_type : str
        User-provided or config-provided experiment type.

    Returns
    -------
    ProblemType
        The corresponding enum indicating the type of problem.
    """
    exp_type = exp_type.lower()
    if exp_type in ["multiobjective", "moo"]:
        return ProblemType.MULTIOBJECTIVE
    if exp_type in ["classification"]:
        return ProblemType.CLASSIFICATION
    return ProblemType.REGRESSION
