from data import HandWritten, CUB, Scene, PIE, Caltech, NUS
from model_train import normal, conflict

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                        help='input batch size for training [default: 200]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--comm-feature-dim', type=int, default=64, metavar='N',
                        help='number of common and specific feature dimensions [default: 64]')
    parser.add_argument('--common-feature-coefficient', type=float, default=1.0, metavar='C',
                        help='loss coefficient for extracting common feature [default: 1.0]')
    parser.add_argument('--specific-feature-coefficient', type=float, default=0.001, metavar='S',
                        help='loss coefficient for extracting specific feature')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='gamma',
                        help='loss coefficient for the consistency of results [default: 1.0]')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()

    dataset = HandWritten()
    # 1.Accuracy for Normal test
    acc = normal(dataset, args)
    # 2.Accuracy for Conflict test
    # acc = conflict(dataset, args)
    print('====> acc: {:.4f}'.format(acc))
