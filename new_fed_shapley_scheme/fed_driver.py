import os
import argparse as ap  



if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Creating Info. for Comp. Shapley.")
    parser.add_argument("--model", type=str, default='linear')
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--rec_grad", type=bool, default=False)
    # parser.add_
    
    # parser.add_argument('--dataset', type=str, default= 'emnist')
    # parser.add_argument('--c_num', type=int, default=10)
    # parser.add_argument('--c_type', type=str, default='same')


    args = parser.parse_args()