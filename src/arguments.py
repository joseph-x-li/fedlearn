import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of rounds of training.") 
    # parser.add_argument("--num_users", type=int, default=100,
    #                     help="Number of users")
    parser.add_argument("--frac", type=float, default=-1,
                        help="The fraction of clients per round. If this is set, cpr is ignored.")
    parser.add_argument("--cpr", type=int, default=-1,
                        help="The number of clients per round.")
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
    # parser.add_argument("--num_classes", type=int, default=10, help="number \
    #                     of classes")
    # parser.add_argument("--optimizer", type=str, default="sgd", help="type \
    #                     of optimizer")
    parser.add_argument("--iid", type=int, default=1,
                        help="Default set to IID. Set to 0 for non-IID.")
    # parser.add_argument("--unequal", type=int, default=0,
    #                     help="whether to use unequal data splits for  \
    #                     non-i.i.d setting (use 0 for equal splits)")
    # parser.add_argument("--verbose", type=int, default=1, help="verbose")
    # parser.add_argument("--seed", type=int, default=1, help="random seed")

    # FedSEM args
    parser.add_argument("--clusters", type=int, default=3,
                        help="Number of FedSEM clusters.")
    parser.add_argument("--sharing", type=int, default=0,
                        help="Set to 1 to enable weight sharing.")

    # FedSEM args
    parser.add_argument("--clusters", type=int, default=3,
                        help="Number of FedSEM clusters.")
    parser.add_argument("--sharing", type=int, default=0,
                        help="Set to 1 to enable weight sharing.")

    args = parser.parse_args()
    if args.dataset_size not in ["small", "large"]:
        raise ValueError("Argument {}")
    return args