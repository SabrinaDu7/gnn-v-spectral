from data.real_world import (
    load_polblogs,
    print_basic_graph_properties,
    extract_largest_connected_component,
)

graph = load_polblogs("data/cache/realworld/polblogs")

print("FULL GRAPH")
print_basic_graph_properties(graph)

print("\nGCC")
gcc = extract_largest_connected_component(graph)
print_basic_graph_properties(gcc)