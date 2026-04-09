from __future__ import annotations

from pathlib import Path

from data.real_world.characterize import basic_graph_properties, extract_largest_connected_component, filter_classes_by_min_size
from data.real_world.loaders import load_polblogs, load_lastfm_asia, load_facebook_residence



def print_props(title: str, props: dict) -> None:
    print(f"\n=== {title} ===")
    keys = [
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
    ]
    for k in keys:
        if k in props:
            print(f"{k}: {props[k]}")


def main() -> None:
    # adjust these roots to your actual raw/processed folders if needed
    polblogs_root = Path("data/cache/realworld_raw/polblogs")
    lastfm_root = Path("data/cache/realworld_raw/lastfm_asia")
    fb_root = Path("data/cache/realworld_raw/facebook_penn94")

    polblogs = load_polblogs(polblogs_root)
    polblogs_gcc = extract_largest_connected_component(polblogs)
    print_props("PolBlogs", basic_graph_properties(polblogs_gcc))

    lastfm = load_lastfm_asia(lastfm_root)
    print_props("LastFM Asia", basic_graph_properties(lastfm))

    fb = load_facebook_residence(fb_root)
    # print_props("Facebook Penn94", basic_graph_properties(fb))

    # optional variants if you want the graph you actually expect to analyze
    # print_props("Facebook Penn94 GCC", basic_graph_properties(fb_gcc))

    fb_filtered = filter_classes_by_min_size(fb, min_size=50)

    fb_final = extract_largest_connected_component(fb_filtered)
    print_props("Facebook Penn94 GCC + min class size 50", basic_graph_properties(fb_final))


if __name__ == "__main__":
    main()