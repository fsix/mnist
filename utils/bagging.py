"""
Bagging
Take a majority vote between all results.

- idea taken from Leo Breiman
"""

from collections import defaultdict
from typing import List, Dict


def bagging(artifacts: List, results: List[Dict]) -> List:
    """Aggregate the results by taking a majority vote for each classification.

    :param artifacts: List of artifacts
    :param results: A list of dictionaries, which map artifact to category
    :return: List of categories
    """
    aggregation = []
    for artifact in artifacts:
        counts = defaultdict(lambda: 0)
        for result in results:
            counts[result[artifact]] += 1
        aggregation.append(max(counts, key=lambda k: counts[k]))
    return aggregation
