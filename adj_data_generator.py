import argparse


def get_sample(TV, A1, A2, A3, N, L1, L2, L3):
    X = [TV]
    if A1 is not None:
        X.append(A1)
    if A2 is not None:
        X.append(A2)
    if A3 is not None:
        X.append(A3)
    X.append(N)

    Y = [TV.upper(), N.upper()]
    if A1 is not None:
        Y.append(L1.upper())
    else:
        Y.append('NA')
    if A2 is not None:
        Y.append(L2.upper())
    else:
        Y.append('NA')
    if A3 is not None:
        Y.append(L3.upper())
    else:
        Y.append('NA')
    return X, Y


def get_voc_full(is_train=True):
    VT = ['push', 'pull', 'raise', 'spin']
    SIZE = ['small', 'large']
    COLOR = ['yellow', 'purple', 'brown', 'blue', 'red', 'gray', 'green', 'cyan']
    if is_train:
        MATERIAL = ['metal', 'plastic']
    else:
        MATERIAL = ['rubber']
    NOUN = ['sphere', 'cylinder', 'cube']
    return VT, SIZE, COLOR, MATERIAL, NOUN


def get_voc_trainB():
    VT = ['push']
    SIZE = ['small']
    COLOR = ['yellow']
    MATERIAL = ['rubber']
    NOUN = ['sphere']
    return VT, SIZE, COLOR, MATERIAL, NOUN


def filter(train, test):
    s = set()
    for entry in train:
        s.add(get_line(entry[0], entry[1]))
    result = []
    for entry in test:
        if get_line(entry[0], entry[1]) not in s:
            result.append(entry)
    return result


def get_data(voc, shuf=True):
    VT, SIZE, COLOR, MATERIAL, NOUN = voc
    result = []
    for v in VT:
        for s in SIZE:
            for c in COLOR:
                for m in MATERIAL:
                    for n in NOUN:
                        result.append(get_sample(v, s, c, m, n, s, c, m))
                        if shuf:
                            result.append(get_sample(v, s, m, c, n, s, c, m))
                            result.append(get_sample(v, c, s, m, n, s, c, m))
                            result.append(get_sample(v, c, m, s, n, s, c, m))
                            result.append(get_sample(v, m, s, c, n, s, c, m))
                            result.append(get_sample(v, m, c, s, n, s, c, m))
    return result


def get_line(X, Y):
    return "IN: " + " ".join(X) + " OUT: " + " ".join(Y)


def main(args):
    trainA = get_data(get_voc_full())
    print('trainA: ', len(trainA))
    if args.hard:
        trainB = get_data(get_voc_trainB(), shuf=False)
    else:
        trainB = get_data(get_voc_full(is_train=False), shuf=False)
    print('trainB: ', len(trainB))
    test = get_data(get_voc_full(is_train=False))
    print('test: ', len(test))
    test = filter(trainA + trainB, test)
    print('filtered test: ', len(test))

    if args.hard:
        train_fn = 'data_adj/train_hard.txt'
        test_fn = 'data_adj/test_hard.txt'
        repeat = round(len(trainA) / 9.)
    else:
        train_fn = 'data_adj/train.txt'
        test_fn = 'data_adj/test.txt'
        repeat = 1
    with open(train_fn, 'w') as f:
        for entry in trainA:
            f.write(get_line(entry[0], entry[1]) + '\n')
        for entry in trainB:
            for _ in range(repeat):
                f.write(get_line(entry[0], entry[1]) + '\n')

    with open(test_fn, 'w') as f:
        for entry in test:
            f.write(get_line(entry[0], entry[1]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention Visualization.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='hard dataset.')
    args = parser.parse_args()
    main(args)
