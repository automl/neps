from __future__ import annotations

import os

import yaml


class SearcherConfigs:
    """
    This class provides methods to access default configuration details
    for NePS optimizers.
    """

    @staticmethod
    def _get_searchers_folder_path() -> str:
        """
        Helper method to get the folder path for default searchers.

        Returns:
            str: The absolute path to the default searchers folder.
        """
        script_directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_directory, "default_searchers")

    @staticmethod
    def get_searchers() -> list[str]:
        """
        List all the searcher names that can be used in neps run.

        Returns:
            list[str]: A list of searcher names.
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()
        searchers = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".yaml"):
                searcher_name = os.path.splitext(file_name)[0]
                searchers.append(searcher_name)

        return searchers

    @staticmethod
    def get_available_algorithms() -> list[str]:
        """
        List all available algorithms used by NePS searchers.

        Returns:
            list[str]: A list of algorithm names.
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()
        prev_algorithms = set()

        for filename in os.listdir(folder_path):
            if filename.endswith(".yaml"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path) as file:
                    searcher_config = yaml.safe_load(file)
                    algorithm = searcher_config["searcher_init"].get("algorithm")
                    if algorithm:
                        prev_algorithms.add(algorithm)

        return list(prev_algorithms)

    @staticmethod
    def get_searcher_from_algorithm(algorithm: str) -> list[str]:
        """
        Get all NePS searchers that use a specific searching algorithm.

        Args:
            algorithm (str): The name of the algorithm needed for the search.

        Returns:
            list[str]: A list of searcher names using the specified algorithm.
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()
        searchers = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".yaml"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path) as file:
                    searcher_config = yaml.safe_load(file)
                    if searcher_config["searcher_init"].get("algorithm") == algorithm:
                        searchers.append(os.path.splitext(filename)[0])

        return searchers

    @staticmethod
    def get_searcher_kwargs(searcher: str) -> str:
        """
        Get the kwargs and algorithm setup for a specific searcher.

        Args:
            searcher (str): The name of the searcher to check the details of.

        Returns:
            str: The raw content of the searcher's configuration
        """
        folder_path = SearcherConfigs._get_searchers_folder_path()

        for filename in os.listdir(folder_path):
            if filename.endswith(".yaml") and filename.startswith(searcher):
                file_path = os.path.join(folder_path, filename)
                with open(file_path) as file:
                    searcher_config = file.read()

        return searcher_config
