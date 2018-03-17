class ExampleParser(object):
    """
    feature extracting, remapping, filtering etc should be done here
    """
    def __call__(self, example):
        raise NotImplementedError()


class YTExampleParser(ExampleParser):
    def __init__(self, factors, label):
        self._factors = factors
        self._label = label

    def __call__(self, example):
        X = [getattr(example, f) for f in self._factors]
        y = getattr(example, self._label)
        return X, y


class TSExampleParser(ExampleParser):
    def __init__(self):
        pass

    def __call__(self, example):
        """
        :param example:
        :return: tuple  = list of features, label
        """
        # We assume here for now, that all the preprocessing has already been done
        splitted = example.split()
        X = splitted[:-1]
        y = splitted[-1]
        return X, y


class BatchIter(object):
    def __init__(self, stream, batch_size, parser):
        self._stream = stream
        self._batch_size = batch_size
        self._parser = parser

    def __next__(self):
        size = 0
        features = []
        labels = []
        example = next(self._stream)

        X, y = self._parser(example)
        features.append(X)
        labels.append(y)
        size += 1
        while size < self._batch_size:
            try:
                example = next(self._stream)
            except StopIteration:
                break
            X, y = self._parser(example)
            features.append(X)
            labels.append(y)
        return features, labels

    def __iter__(self):
        return self


