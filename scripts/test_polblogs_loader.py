from data.real_world.loaders import load_polblogs, save_real_world_graph

raw_dir = "data/cache/realworld_raw/polblogs"

graph = load_polblogs(raw_dir)

print("Loaded graph successfully.")
print(graph.metadata)
print(graph.edges.head())
print(graph.labels[:10])

save_real_world_graph(graph, raw_dir)
print("Saved processed graph files.")