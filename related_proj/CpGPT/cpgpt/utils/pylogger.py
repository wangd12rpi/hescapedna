import logging
from collections.abc import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """Multi-GPU-friendly python command line logger.

    Provides logging functionality that works correctly across multiple GPU processes
    by prefixing log messages with their rank.

    Attributes:
        rank_zero_only (bool): Whether to restrict logging to rank zero process

    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize multi-GPU-friendly logger.

        Args:
            name (str, optional): Logger name. Defaults to __name__
            rank_zero_only (bool, optional): Whether to force logging only on rank zero.
                Defaults to False
            extra (Optional[Mapping[str, object]], optional): Additional contextual information.
                Defaults to None

        Note:
            Uses logging.LoggerAdapter as base class for contextual information

        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self,
        level: int,
        msg: str,
        rank: int | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Log a message with rank prefix.

        Delegates logging after adding process rank prefix to message.

        Args:
            level (int): Logging level (from logging module)
            msg (str): Message to log
            rank (Optional[int], optional): Specific rank to log from. Defaults to None
            *args: Additional positional arguments for logging function
            **kwargs: Additional keyword arguments for logging function

        Raises:
            RuntimeError: If rank_zero_only.rank is not set before use

        Note:
            - If rank_zero_only is True, only logs on rank 0
            - If rank is specified, only logs on that rank
            - Otherwise logs on all ranks with rank prefix

        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                msg = "The `rank_zero_only.rank` needs to be set before use"
                raise RuntimeError(msg)
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            elif rank is None or current_rank == rank:
                self.logger.log(level, msg, *args, **kwargs)
