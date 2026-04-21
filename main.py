from checkers.graph.graph import checkers_graph

def main():
    # Draw and save the graph
    img = checkers_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)
    print("Graph saved as graph.png")

    # Run the graph with initial state
    result = checkers_graph.invoke({})
    print("Done:", result)

if __name__ == "__main__":
    main()