import argparse
import matplotlib.pyplot as plt
import numpy as np
from .run import Options
from .utils.metric import RMSE, MAPE

def parse_args():
    parser = argparse.ArgumentParser(description="Test the model and calculate metrics")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model zip file")
    parser.add_argument("--test_time", type=str, required=True, help="Test time in the format YYYY-MM-DD HH:MM:SS")
    return parser.parse_args()

def main():
    args = parse_args()

    model = Options().get_model(args.model_path)
    
    # get data
    stream_data, indicator_data, _ = model.get_data(args.test_time)  

    predicted, ground_truth = model.test_single(args.test_time)
    
    # metric
    print("RMSE: ", RMSE(predicted, ground_truth))
    print("MAPE: ", MAPE(predicted, ground_truth))
    
    plt.figure(figsize=(10, 6))
    
    x_stream = np.arange(288)
    x_ground_truth = np.arange(288, 288+288)
    x_predicted = np.arange(288, 288+288)
                        
    plt.plot(x_stream, stream_data[-288:], label='Stream Data', color='black')
    plt.plot(x_ground_truth, ground_truth, label='Ground Truth', color='black')
    plt.plot(x_predicted, predicted, label='Predicted', color='b') 
    max_value = max(max(ground_truth), max(predicted), max(stream_data[-288:]))
    plt.ylim(0, max_value * 1.2)
    plt.savefig('output.png')
    plt.show(block=True)
    print("Figure saved in output.png!")

if __name__ == "__main__":
    main()

