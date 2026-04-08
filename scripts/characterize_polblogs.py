from data.real_world import load_polblogs, print_basic_graph_properties

graph = load_polblogs("data/cache/realworld/polblogs")
print_basic_graph_properties(graph)