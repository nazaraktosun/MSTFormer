from collections import defaultdict,deque

def resolve_feature_dependencies(feature_specs: list[dict]) -> list[str]:
    """
    Topologically sorts features based on their dependencies.

    Args:
        feature_specs: A list of feature specifications. Each spec is a dictionary
                       that must contain a "name" and a "deps" list.

    Returns:
        A list of feature names in the correct execution order.

    Raises:
        ValueError: If a feature lists a dependency that doesn't exist,
                    or if a circular dependency is detected.
    """

    feature_map = {spec["name"]: spec for spec in feature_specs}
    dependency_graph = defaultdict(list)
    in_degree = {name :0 for name in feature_map}

    for name,spec in feature_map.items():
        for dep in spec.get("deps",[]):
            if dep not in feature_map:
                raise ValueError (f"Feature {name} has an undefined dependency : '{dep}'")
            dependency_graph[dep].append(name)
            in_degree[name] += 1

    queue = deque(name for name, degree in in_degree.items() if degree == 0)