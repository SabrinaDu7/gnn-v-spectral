from data.real_world import load_lastfm_asia, print_basic_graph_properties

graph = load_lastfm_asia("data/cache/realworld_raw/lastfm_asia")

print("Loaded LastFM Asia successfully.")
print(graph.metadata)
print(graph.edges.head())
print(graph.labels[:10])
print(graph.features[:2, :10])

print("\nCharacterization")
print_basic_graph_properties(graph)