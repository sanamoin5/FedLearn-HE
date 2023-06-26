import argparse


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument(
        "--fed_algo",
        type=str,
        default="FedAvg",
        help="Specify federated aggregation algorithm to use(FedAvg, FedProx)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="number of rounds of training ()Default: 10",
    )
    parser.add_argument(
        "--num_clients", type=int, default=2, help="number of clients (Default: 2)"
    )
    parser.add_argument(
        "--client_weights",
        nargs="+",
        type=float,
        default=None,
        help="weights for each of the client",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="the number of local epochs (Default: 10)",
    )
    parser.add_argument("--local_bs", type=int, default=64, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )

    parser.add_argument("--gpu_id", default=None, help="Use to set a gpu id")

    # Homomorphic encryption arguments
    parser.add_argument(
        "--he_scheme_name",
        type=str,
        default="ckks",
        help="define the homomorphic encryption scheme to be used, (ckks, paillier or bfv)",
    )
    parser.add_argument(
        "--bits_scale",
        type=int,
        default=40,
        help="the scale to be used to encode tensor values. CKKSTensor will use the global_scale "
        "provided by the context if it is set to None.",
    )
    parser.add_argument(
        "--poly_modulus_degree",
        type=int,
        default=8192,
        help="The degree of the polynomial modulus, must be a power of two",
    )
    parser.add_argument(
        "--coeff_mod_bit_sizes",
        nargs="+",
        type=int,
        default=[40, 20, 40],
        help="List of bit size for each coefficient modulus. Can be an empty list for BFV, a default "
        "value will be given.",
    )
    parser.add_argument(
        "--global_scale", type=int, default=40, help="the scaling factor"
    )

    parser.add_argument(
        "--create_galois_keys",
        dest="create_galois_keys",
        action="store_true",
        help="Enables generation of public keys needed to perform encrypted vector rotation operations"
        " on batched ciphertexts (default not set)",
    )
    parser.add_argument(
        "--no_galois_keys",
        dest="create_galois_keys",
        action="store_false",
        help="Disables generation of public keys needed to perform encrypted vector rotation operations"
        " on batched ciphertexts (default set)",
    )
    parser.set_defaults(create_galois_keys=False)

    parser.add_argument(
        "--plain_modulus",
        type=int,
        default=1032193,
        help="The plaintext modulus. Should not be passed when the scheme is CKKS.",
    )

    # other arguments
    parser.add_argument(
        "--train_with_encryption",
        dest="train_without_encryption",
        action="store_true",
        help="Set if model wants to be tested with encryption, default set",
    )
    parser.add_argument(
        "--train_without_encryption",
        dest="train_without_encryption",
        action="store_false",
        help="Set if model does not want to be tested with encryption (default not set).",
    )
    parser.set_defaults(train_without_encryption=True)

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="name \
                        of dataset(mnist, cifar-10)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number \
                        of classes",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="type \
                        of optimizer",
    )
    parser.add_argument(
        "--iid", type=int, default=1, help="Default set to IID. Set to 0 for non-IID."
    )
    parser.add_argument(
        "--unequal",
        type=int,
        default=0,
        help="whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)",
    )

    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args()
    return args
