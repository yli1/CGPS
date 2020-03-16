import argparse
import random
import editdistance
import os
import numpy as np

from data_generator import Tokenizer
from data_generator import get_data_generator
from model import get_model


def visualization(i, a, b, c, d, e, directory1):
    with open(directory1 + '/' + str(i) + '.txt', 'w') as f:
        a = a + ["EOS_X"]
        c = c + ["EOS_Y"]
        f.write(str(i) + ' ')
        f.write(' '.join(a))
        f.write('\n')
        f.write('switch ')
        f.write(' '.join(str(x[0]) for x in e))
        f.write('\n')
        for p, q in zip(c, d):
            f.write(p + ' ')
            f.write(' '.join(str(x) for x in q))
            f.write('\n')


def evaluation(test_X, test_Y, prediction, attention, switch, act, fn):
    id2act = {i: a for a, i in act.items()}
    actions = []
    for pred in prediction:
        acts = []
        for id in pred:
            if id == 0:
                break
            acts.append(id2act[id])
        actions.append(acts)

    directory = fn + "/attention"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(fn + '/output.txt', 'w') as f:
        for i, (a, b, c, d, e) in enumerate(zip(test_X, test_Y, actions, attention, switch)):
            ed = editdistance.eval(b, c)
            wer = ed / float(len(b))
            f.write(str(i) + '\t')
            f.write(str(len(b)) + '\t')
            f.write(str(len(c)) + '\t')
            f.write(str(ed) + '\t')
            f.write(str(wer))
            f.write('\n')
            f.write(' '.join(a))
            f.write('\n')
            f.write(' '.join(str(x[0]) for x in e))
            f.write('\n')
            f.write(' '.join(b))
            f.write('\n')
            f.write(' '.join(c))
            f.write('\n\n')
            visualization(i, a, b, c, d, e, directory)


def process(args):
    # prepare data
    dg = get_data_generator(args.data_name, args)
    train_X, train_Y = dg.get_train_data()
    test_X, test_Y = dg.get_test_data()

    if args.use_start_symbol:
        train_X = [['S'] + x for x in train_X]
        test_X = [['S'] + x for x in test_X]

    ori_test_X, ori_test_Y = test_X, test_Y

    # Tokenize
    tokenizer = Tokenizer(args)
    samples, dicts, lengths, maxs = tokenizer.initialize(
        train_X, train_Y, test_X, test_Y)
    train_X, train_Y, test_X, test_Y = samples
    voc, act = dicts
    train_X_len, train_Y_len, test_X_len, test_Y_len = lengths

    if args.remove_x_eos:
        train_X_len = [x - 1 for x in train_X_len]
        test_X_len = [x - 1 for x in test_X_len]

    max_input, max_output = maxs

    args.input_length = max_input
    args.output_length = max_output

    # prepare model
    model = get_model(args.model_name, args)
    model.initialize(len(voc) + 1, len(act) + 1)
    model.train(train_X, train_Y, train_X_len, train_Y_len)

    model.test(train_X, train_Y, train_X_len, train_Y_len, "Train w. noise", noise_weight=args.noise_weight)
    model.test(train_X, train_Y, train_X_len, train_Y_len, "Train w.o. noise")

    model.test(test_X, test_Y, test_X_len, test_Y_len, "Test w. noise", noise_weight=args.noise_weight)
    prediction, attention, switch, sent_acc = model.test(test_X, test_Y, test_X_len, test_Y_len, "Test w.o. noise")
    evaluation(ori_test_X, ori_test_Y, prediction, attention, switch, act, 'logs/' + args.experiment_id)
    print("Final sentence accuracy:", str(100 * sent_acc) + '%')


def main(args):
    seed = args.random_seed
    random.seed(seed)
    if args.random_random:
        np.random.seed(random.randint(2, 1000))
    else:
        np.random.seed(seed)

    # organizing parameters
    if args.remove_noise:
        args.noise_weight = 0.0
    if args.function_embedding_size <= 0:
        args.function_embedding_size = args.embedding_size

    process(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compositional Instructions.')
    parser.add_argument('--experiment_id', type=str, default='default',
                        help='experiment ID')
    parser.add_argument('--model_name', type=str, default='transformer',
                        help='model name')
    parser.add_argument('--print_output', action='store_true', default=False,
                        help='Linear max.')
    parser.add_argument('--simple_data', action='store_true', default=False,
                        help='use simple data.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--learning_rate', type=float, default=0.3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch_size')
    parser.add_argument('--shuffle_batch', action='store_true', default=False,
                        help='shuffle batch.')
    parser.add_argument('--random_batch', action='store_true', default=False,
                        help='random batch.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='epochs')
    parser.add_argument('--data_name', type=str, default='scan',
                        help='name of data set')
    parser.add_argument('--train_file', type=str,
                        default='SCAN/add_prim_split/tasks_train_addprim_jump.txt',
                        help='train file name')
    parser.add_argument('--test_file', type=str,
                        default='SCAN/add_prim_split/tasks_test_addprim_jump.txt',
                        help='test file name')
    parser.add_argument('--switch_temperature', type=float, default=1.0,
                        help='switch temperature')
    parser.add_argument('--attention_temperature', type=float, default=10.0,
                        help='attention temperature')
    parser.add_argument('--num_units', type=int, default=16,
                        help='num units')
    parser.add_argument('--bidirectional_encoder', action='store_true', default=False,
                        help='bidirectional encoder.')
    parser.add_argument('--max_gradient_norm', type=float, default=-1.0,
                        help='max gradient norm')
    parser.add_argument('--decay_steps', type=int, default=-1,
                        help='decay steps')
    parser.add_argument('--use_input_length', action='store_true', default=False,
                        help='use input length.')
    parser.add_argument('--use_embedding', action='store_true', default=False,
                        help='use embedding.')
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='embedding size')
    parser.add_argument('--function_embedding_size', type=int, default=-1,
                        help='function embedding size')
    parser.add_argument('--reg_coe', type=float, default=-1.0,
                        help='regularization coeficient')
    parser.add_argument('--macro_switch_reg_coe', type=float, default=-1.0,
                        help='macro switch regularization coeficient')
    parser.add_argument('--relu_switch', action='store_true', default=False,
                        help='relu switch')
    parser.add_argument('--use_start_symbol', action='store_true', default=False,
                        help='use start symbol')
    parser.add_argument('--content_noise', action='store_true', default=False,
                        help='add noise to content')
    parser.add_argument('--content_noise_coe', type=float, default=-1.0,
                        help='noise regularization coeficient')
    parser.add_argument('--sample_wise_content_noise', action='store_true', default=False,
                        help='sample-wise noise regularization')
    parser.add_argument('--noise_weight', type=float, default=1.0,
                        help='noise weight')
    parser.add_argument('--remove_noise', action='store_true', default=False,
                        help='remove noise')
    parser.add_argument('--function_noise', action='store_true', default=False,
                        help='add noise to function')
    parser.add_argument('--remove_x_eos', action='store_true', default=False,
                        help='remove x eos')
    parser.add_argument('--masked_attention', action='store_true', default=False,
                        help='masked attention')
    parser.add_argument('--remove_switch', action='store_true', default=False,
                        help='remove switch')
    parser.add_argument('--use_entropy_reg', action='store_true', default=False,
                        help='use entropy reg')
    parser.add_argument('--random_random', action='store_true', default=False,
                        help='random_random')
    parser.add_argument('--single_representation', action='store_true', default=False,
                        help='single representation')
    parser.add_argument('--use_decoder_input', action='store_true', default=False,
                        help='single representation')
    parser.add_argument('--output_embedding_size', type=int, default=8,
                        help='output embedding size')
    parser.add_argument('--use_l1_norm', action='store_true', default=False,
                        help='single representation')
    parser.add_argument('--remove_prediction_bias', action='store_true', default=False,
                        help='remove prediction bias')
    parser.add_argument('--clip_by_norm', action='store_true', default=False,
                        help='clip by norm instead of global norm.')
    args = parser.parse_args()

    main(args)
