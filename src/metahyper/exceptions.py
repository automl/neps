class MissingDependencyError(Exception):
    """Used whenever an optional dependancy is missing."""

    def __init__(
        self,
        *,
        libname: str,
        dep: str,
        cause: Exception,
        install_group: str | None = None,
    ):
        """Initialize the exception.

        Args:
            libname (str): The name of the library that is missing the dependency.
            dep (str): The name of the dependency that is missing.
            install_group (str, optional): The name of the dependency group to install.
            cause (Exception): The exception that caused the error.
        """
        super().__init__(libname, dep, cause, install_group)
        self.libname = libname
        self.dep = dep
        self.install_group = install_group
        self.__cause__ = cause  # This is what `raise a from b` does

    def __str__(self) -> str:
        msg = f"Required dependency ({self.dep}) is missing for this optional feature."
        if self.install_group is not None:
            msg += (
                f"Please install with {self.libname}[{self.install_group}] "
                f"to be able to use all the optional features. Otherwise, "
                f" just install ({self.dep})"
            )
        return msg
