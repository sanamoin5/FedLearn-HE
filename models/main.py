import torch
from args_parser import args_parser
import time
from client import Client

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # call client
    client = Client()
    client.train_model(test=True, save=False, plot=True)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))