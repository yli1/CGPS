def get_data_generator(name, args):
    if name == 'fewshot':
        return FewShotDataGanerator(args)
    elif name == 'scan':
        return SCANGanerator(args)
    elif name == 'toy':
        return ToyDataGanerator(args)
    else:
        raise ValueError("Data generator name is not defined: " + name)


class FewShotDataGanerator(object):
    def __init__(self, args):
        self.args = args

    def get_train_data(self):
        data = []

        # Primitives
        data.append(('dax', 'R'))
        data.append(('lug', 'B'))
        data.append(('wif', 'G'))
        data.append(('zup', 'Y'))

        # Function 1
        data.append(('lug fep', 'BBB'))
        data.append(('dax fep', 'RRR'))

        # Function 2
        data.append(('lug blicket wif', 'BGB'))
        data.append(('wif blicket dax', 'GRG'))

        # Function 3
        data.append(('lug kiki wif', 'GB'))
        data.append(('dax kiki lug', 'BR'))

        if not self.args.simple_data:
            # Function compositions
            data.append(('lug fep kiki wif', 'GBBB'))
            data.append(('wif kiki dax blicket lug', 'RBRG'))
            data.append(('lug kiki wif fep', 'GGGB'))
            data.append(('wif blicket dax kiki lug', 'BGRG'))

        X = [x[0].split() for x in data]
        Y = [list(x[1]) for x in data]

        return X, Y

    def get_test_data(self):
        data = []

        # Function 1
        data.append(('zup fep', 'YYY'))

        # Function 2
        data.append(('zup blicket lug', 'YBY'))
        data.append(('dax blicket zup', 'RYR'))

        # Function 3
        data.append(('zup kiki dax', 'RY'))
        data.append(('wif kiki zup', 'YG'))

        if not self.args.simple_data:
            # Function compositions
            data.append(('zup fep kiki lug', 'BYYY'))
            data.append(('wif kiki zup fep', 'YYYG'))
            data.append(('lug kiki wif blicket zup', 'GYGB'))
            data.append(('zup blicket wif kiki dax fep', 'RRRYGY'))
            data.append(('zup blicket zup kiki zup fep', 'YYYYYY'))

        X = [x[0].split() for x in data]
        Y = [list(x[1]) for x in data]

        return X, Y


class ToyDataGanerator(object):
    def __init__(self, args):
        self.args = args

    def get_train_data(self):
        data = []

        # Primitives
        data.append(('small apple', 'ASN'))
        data.append(('small melon', 'MSN'))
        data.append(('large apple', 'ALN'))
        data.append(('large melon', 'MLN'))
        data.append(('green apple', 'ANG'))
        data.append(('red apple', 'ANR'))
        data.append(('red melon', 'MNR'))

        X = [x[0].split() for x in data]
        Y = [list(x[1]) for x in data]

        return X, Y

    def get_test_data(self):
        data = []

        # Function 1
        data.append(('green melon', 'MNG'))

        X = [x[0].split() for x in data]
        Y = [list(x[1]) for x in data]

        return X, Y


class SCANGanerator(object):
    def __init__(self, args):
        self.args = args

    def load(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        input_list = []
        output_list = []
        for line in lines:
            _, left, right = line.split(':')
            input_list.append(left.strip().split()[:-1])
            output_list.append(right.strip().split())
        return input_list, output_list

    def get_train_data(self):
        return self.load(self.args.train_file)

    def get_test_data(self):
        return self.load(self.args.test_file)


class Formarter(object):
    def __init__(self, args):
        self.args = args

    def get_dict(self, seqs):
        s = set()
        for seq in seqs:
            for elem in seq:
                s.add(elem)
        return {e: i + 1 for i, e in enumerate(s)}

    def convert_sequence(self, seqs, dic):
        result = []
        for seq in seqs:
            a = []
            for elem in seq:
                if elem not in dic:
                    unk = '<unk>'
                    if unk not in dic:
                        dic[unk] = len(dic) + 1
                    a.append(dic[unk])
                else:
                    a.append(dic[elem])
            result.append(a)
        return result

    def padding(self, seqs, el, pad=0):
        lengths = []
        for seq in seqs:
            lengths.append(len(seq) + 1)
            for _ in range(el - len(seq)):
                seq.append(pad)
        return seqs, lengths

    def initialize_basic(self, X, Y, X_test, Y_test):
        voc = self.get_dict(X)
        act = self.get_dict(Y)

        x_out = self.convert_sequence(X, voc)
        y_out = self.convert_sequence(Y, act)
        x_test_out = self.convert_sequence(X_test, voc)
        y_test_out = self.convert_sequence(Y_test, act)
        return x_out, y_out, x_test_out, y_test_out, voc, act

    def get_maximum_length(self, train, test):
        train_max = max([len(x) for x in train])
        test_max = max([len(x) for x in test])
        return max(train_max, test_max) + 1

    def initialize(self, X, Y, X_test, Y_test):
        X, Y, X_test, Y_test, voc, act = self.initialize_basic(
            X, Y, X_test, Y_test)
        max_input = self.get_maximum_length(X, X_test)
        max_output = self.get_maximum_length(Y, Y_test)
        X, X_len = self.padding(X, max_input)
        Y, Y_len = self.padding(Y, max_output)
        X_test, X_test_len = self.padding(X_test, max_input)
        Y_test, Y_test_len = self.padding(Y_test, max_output)
        samples = X, Y, X_test, Y_test
        dicts = voc, act
        lengths = X_len, Y_len, X_test_len, Y_test_len
        maxs = max_input, max_output
        return samples, dicts, lengths, maxs
