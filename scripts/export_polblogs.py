from data.real_world import (
    load_polblogs,
    extract_largest_connected_component,
    save_real_world_graph,
    print_basic_graph_properties,
)

RAW_DIR = "data/cache/realworld/polblogs"

FULL_OUT_DIR = "data/cache/realworld/polblogs/full"
GCC_OUT_DIR = "data/cache/realworld/polblogs/gcc"


def main() -> None:
    full_graph = load_polblogs(RAW_DIR)
    gcc_graph = extract_largest_connected_component(full_graph)

    print("FULL GRAPH")
    print_basic_graph_properties(full_graph)

    print("\nGCC GRAPH")
    print_basic_graph_properties(gcc_graph)

    save_real_world_graph(full_graph, FULL_OUT_DIR)
    save_real_world_graph(gcc_graph, GCC_OUT_DIR)

    print("\nSaved processed PolBlogs graphs:")
    print(f"  Full: {FULL_OUT_DIR}")
    print(f"  GCC:  {GCC_OUT_DIR}")


if __name__ == "__main__":
    main()