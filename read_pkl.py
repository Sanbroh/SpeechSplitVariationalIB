#!/usr/bin/env python3
import pickle
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_pickle.py <path_to_pickle_file>")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        print("Content of the pickle file:")
        print(data)
    except Exception as e:
        print(f"Error reading pickle file: {e}")

if __name__ == "__main__":
    main()

