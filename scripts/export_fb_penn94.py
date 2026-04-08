from pathlib import Path

from data.real_world import (
    load_facebook_residence,
    filter_classes_by_min_size,
    extract_largest_connected_component,
    print_basic_graph_properties,
    save_real_world_graph,
)

RAW_DIR = "data/cache/realworld/facebook_penn94"
OUT_DIR = "data/cache/realworld/facebook_penn94/min50_gcc"


def main() -> None:
    graph = load_facebook_residence(RAW_DIR, campus_name="Penn94")

    print("ORIGINAL FILTERED GRAPH")
    print_basic_graph_properties(graph)

    min50 = filter_classes_by_min_size(graph, min_size=50)
    print("\nMIN50")
    print_basic_graph_properties(min50)

    min50_gcc = extract_largest_connected_component(min50)
    print("\nMIN50 GCC")
    print_basic_graph_properties(min50_gcc)

    save_real_world_graph(min50_gcc, OUT_DIR)

    print("\nSaved processed Facebook Penn94 graph:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()