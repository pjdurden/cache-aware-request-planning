def pareto_plane(points, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for p in points:
        size = 50 + p.local_cost * 5000
        ax.scatter(p.server_cost, p.quality_cost, s=size)
        ax.annotate(p.method, (p.server_cost, p.quality_cost))
    ax.set_xlabel("server cost (USD)")
    ax.set_ylabel("quality cost")
    ax.set_title("Cost vs quality (marker size = local compute)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
