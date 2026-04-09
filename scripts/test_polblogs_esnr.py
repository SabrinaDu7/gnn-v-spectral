from data.real_world.loaders import load_polblogs
from data.real_world.characterize import basic_graph_properties

raw_dir = "data/cache/realworld_raw/polblogs"

graph = load_polblogs(raw_dir)
props = basic_graph_properties(graph)

print("PolBlogs ESNR inspection")
for key in [
    "graph_id",
    "dataset",
    "n_nodes",
    "n_edges",
    "num_classes",
    "avg_degree",
    "density",
    "num_connected_components",
    "largest_component_fraction",
    "esnr",
    "esnr_n_outliers",
    "esnr_outlier_mass",
    "esnr_converged",
    "esnr_iterations",
]:
    print(f"{key}: {props[key]}")