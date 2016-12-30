from model import DTN
from solver import Solver

def main():
    model = DTN()
    solver = Solver(model, num_epoch=10, svhn_path='svhn/', model_save_path='model/', log_path='log/')
    solver.train()
    

if __name__ == "__main__":
    main()