import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Learning Arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of rounds of training.") 
    parser.add_argument("--local_ep", type=int, default=10,
                        help="The number of local epochs.")
    parser.add_argument("--local_bs", type=int, default=10,
                        help="Local batch size.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Local learning rate.")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="Local SGD momentum.")

    # data arguments
    parser.add_argument("--dataset", type=str, default="femnist", help="Name of dataset")
    parser.add_argument("--dataset_size", type=str, default="small", 
                        help="Dataset size; choose from {small, large}.")
    parser.add_argument("--iid", type=int, default=1,
                        help="Default set to IID. Set to 0 for non-IID.")

    # FedAvg Arguments
    parser.add_argument("--frac", type=float, default=-1,
                        help="The fraction of clients per round. If this is set, cpr is ignored.")
    parser.add_argument("--cpr", type=int, default=-1,
                        help="The number of clients per round.")

    # FedSEM args
    parser.add_argument("--clusters", type=int, default=3,
                        help="Number of FedSEM clusters.")

    # Sampling and Failure args
    parser.add_argument("--sample_dist", type=str, default="uniform",
                        help="Client availability distribution. Choose from {uniform, sigmoid}.")
    parser.add_argument("--sigm_domain", type=float, default=2,
                        help="Sample using sigmoid in the domain [-x, x]. Default to [-2, 2].")
    parser.add_argument("--sharing", type=int, default=0,
                        help="Set to 1 to enable weight sharing.")
    
    # CFL args
    parser.add_argument("--cfl_e1", type=float, default=0,
                        help="epsilon_1 hyperparameter. Cluster splitting occurs when weight update magnitude below this value.")
    parser.add_argument("--cfl_e2", type=float, default=0,
                        help="epsilon_2 hyperparameter. Cluster splitting occurs max weight update is above this value.")

    # CFL args that I made to make it better
    parser.add_argument("--cfl_local_epochs", type=int, default=1,
                        help="Number of local epochs when perform the CFL step. Defaults to 1 (as in paper).")
    parser.add_argument("--cfl_split_every", type=int, default=10,
                        help="Try to split clusters every X global epochs. Defaults to 10.")
    parser.add_argument("--cfl_min_size", type=int, default=10,
                        help="Minimum size of a cluster in CLF. Defaults to 10.")
    parser.add_argument("--cfl_wsharing", type=int, default=0,
                        help="Set to 1 to enable CONV weight sharing. Defaults to 0.")

    args = parser.parse_args()
    if args.dataset_size not in ["small", "large"]:
        raise ValueError("Argument {}")
    return args