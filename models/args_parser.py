import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=2,
                        help="number of users: K")
    parser.add_argument('--epochs', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_id', default=None, help="Use to set a gpu id")

    # Homomorphic encryption arguments
    parser.add_argument('--he_scheme_name', type=str, default='ckks',
                        help='define the homomorphic encryption scheme to be used, (ckks, paillier or bfv)')
    parser.add_argument('--bits_scale', type=int, default=40,
                        help='the scale to be used to encode tensor values. CKKSTensor will use the global_scale '
                             'provided by the context if it is set to None.')
    parser.add_argument('--poly_modulus_degree', type=int, default=8192,
                        help='The degree of the polynomial modulus, must be a power of two')
    parser.add_argument('--coeff_mod_bit_sizes', type=list, default=[40, 20, 40],
                        help='List of bit size for each coefficient modulus. Can be an empty list for BFV, a default '
                             'value will be given.')
    parser.add_argument('--global_scale', type=int, default=40,
                        help='the scaling factor')
    parser.add_argument('--create_galois_keys', type=bool, default=False,
                        help='Enables generation of public keys needed to perform encrypted vector rotation operations '
                             'on batched ciphertexts.')
    parser.add_argument('--plain_modulus', type=int, default=1032193,
                        help='The plaintext modulus. Should not be passed when the scheme is CKKS.')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset(mnist, cifar-10)")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')

    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
