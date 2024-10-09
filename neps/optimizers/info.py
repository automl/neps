from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).parent.resolve()


class SearcherConfigs:
    """This class provides methods to access default configuration details
    for NePS optimizers.
    """

    @staticmethod
    def _get_searchers_folder_path() -> Path:
        """Helper method to get the folder path for default searchers.

        Returns:
            str: The absolute path to the default searchers folder.
        """
        return HERE / "default_searchers"

    @staticmethod
    def get_searchers() -> list[str]:
        """List all the searcher names that can be used in neps run.

        Returns:
            list[str]: A list of searcher names.
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()
        searchers = []

        for file in folder_path.iterdir():
            if file.suffix == ".yaml":
                searchers.append(file.stem)

        return searchers

    @staticmethod
    def get_available_algorithms() -> list[str]:
        """List all available algorithms used by NePS searchers.

        Returns:
            list[str]: A list of algorithm names.
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()
        prev_algorithms = set()

        for file in folder_path.iterdir():
            if file.suffix == ".yaml":
                with file.open("r") as f:
                    searcher_config = yaml.safe_load(f)
                    algorithm = searcher_config.get("strategy")
                    if algorithm:
                        prev_algorithms.add(algorithm)

        return list(prev_algorithms)

    @staticmethod
    def get_searcher_from_algorithm(algorithm: str) -> list[str]:
        """Get all NePS searchers that use a specific searching algorithm.

        Args:
            algorithm (str): The name of the algorithm needed for the search.

        Returns:
            list[str]: A list of searcher names using the specified algorithm.
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()
        searchers = []

        for file in folder_path.iterdir():
            if file.suffix == ".yaml":
                with file.open("r") as f:
                    searcher_config = yaml.safe_load(f)
                    if searcher_config.get("strategy") == algorithm:
                        searchers.append(file.stem)

        return searchers

    @staticmethod
    def get_searcher_kwargs(searcher: str) -> str:
        """Get the kwargs and algorithm setup for a specific searcher.

        Args:
            searcher (str): The name of the searcher to check the details of.

        Returns:
            str: The raw content of the searcher's configuration
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()

        for file in folder_path.iterdir():
            if file.suffix == ".yaml" and file.stem.startswith(searcher):
                return file.read_text()

        raise FileNotFoundError(
            f"Searcher {searcher} not found in default searchers folder."
        )
