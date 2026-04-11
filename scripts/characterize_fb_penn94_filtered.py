from data.real_world import (
    load_facebook_residence,
    print_basic_graph_properties,
    filter_classes_by_min_size,
    extract_largest_connected_component,
)

graph = load_facebook_residence(
    "data/cache/realworld/facebook_penn94",
    campus_name="Penn94",
)

print("ORIGINAL FILTERED GRAPH")
print_basic_graph_properties(graph)

print("\nMIN50")
min50 = filter_classes_by_min_size(graph, min_size=50)
print_basic_graph_properties(min50)

print("\nMIN50 GCC")
min50_gcc = extract_largest_connected_component(min50)
print_basic_graph_properties(min50_gcc)