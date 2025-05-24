from model.predict_graph import predict_graph
from model.graph_solver import min_fuel_cost
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to graph image")
    args = parser.parse_args()

    predicted_num_nodes, predicted_adj_matrix = predict_graph(args.image_path)

    print("Predicted Nodes:", predicted_num_nodes)
    print("Adjacency Matrix:")
    for row in predicted_adj_matrix:
        print(row)

    result = min_fuel_cost(predicted_adj_matrix, predicted_num_nodes)
    print("Minimum Fuel Cost:", result)
