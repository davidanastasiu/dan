import argparse
import matplotlib.pyplot as plt
from .run import Options

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
    p = model.predict(args.test_time, stream_data, indicator_data)
    
    # write to file
    with open("predict.txt", "w") as file:
        file.write(args.test_time + ": ".join(str(round(x, 2)) for x in p) + "\n")
        
    print("Results saved to predict.txt!")
        
if __name__ == "__main__":
    main()

