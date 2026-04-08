from data.real_world import load_facebook_residence, print_basic_graph_properties

graph = load_facebook_residence(
    "data/cache/realworld/facebook_penn94",
    campus_name="Penn94",
)

print("Loaded Facebook Penn94 successfully.")
print(graph.metadata)
print(graph.edges.head())
print(graph.labels[:10])
print(graph.features[:2, :20])

print("\nCharacterization")
print_basic_graph_properties(graph)