from data.real_world import (
    load_lastfm_asia,
    print_basic_graph_properties,
    save_real_world_graph,
)

RAW_DIR = "data/cache/realworld/lastfm_asia"
FULL_OUT_DIR = "data/cache/realworld/lastfm_asia/full"

def main() -> None:
    graph = load_lastfm_asia(RAW_DIR)

    print("FULL GRAPH")
    print_basic_graph_properties(graph)

    save_real_world_graph(graph, FULL_OUT_DIR)

    print("\nSaved processed LastFM Asia graph:")
    print(f"  Full: {FULL_OUT_DIR}")

if __name__ == "__main__":
    main()