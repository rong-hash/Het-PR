__author__ = 'Zhirong Chen'



import sys
import numpy as np
from personalizedModel import PersonalizedThroughMixedModel

def main():
    if len(sys.argv) != 5:
        print("Usage: python run.py <X.npy> <label.npy> <C.npy> <regression model>")
        return

    # Load the data
    X = np.load(sys.argv[1])
    label = np.load(sys.argv[2])
    C = np.load(sys.argv[3])

    # Select the regression model
    regression_model = sys.argv[4]

    # Initialize the model with your desired parameters
    model = PersonalizedThroughMixedModel(mode_regressionModel=regression_model, regWeight=0)

    # Fit the model
    B, P = model.fit(X, label, C)

    # Save B and P
    np.save("B.npy", B)
    np.save("pvalue.npy", P)

    print("B.npy and pvalue.npy have been saved.")

if __name__ == "__main__":
    main()
