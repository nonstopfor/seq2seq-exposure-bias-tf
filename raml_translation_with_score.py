import cotk
from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader.dataloader import *
from collections import Counter
import numpy as np
from itertools import chain


class Score(DataField):
    def get_next(self, dataset):
        r"""read text and returns the next label(integer). Note that it may raise StopIteration.

        Args:{DataField.GET_NEXT_ARG}

        Examples:
            >>> dataset = iter(["1\n", "0\n"])
            >>> field = Label()
            >>> field.get_next(dataset)
            1
            >>> field.get_next(dataset)
            0
        """
        score = next(dataset)
        return float(score.strip())

    def _map_fun(self, element, convert_ids_to_tokens=None):
        """
        Returns the element itself.

        Args:
            element: An element of a dataset.
            convert_ids_to_tokens: It's useless. This argument exists, just to keep the signature the same as that of super class.
        """
        return element


class TranslationWithScore(cotk.dataloader.SingleTurnDialog):
    @cotk._utils.hooks.hook_dataloader
    def __init__(self, file_id, min_vocab_times, \
                 max_sent_length, invalid_vocab_times, \
                 tokenizer, remains_capital
                 ):
        super().__init__(file_id, min_vocab_times, \
                         max_sent_length, invalid_vocab_times, \
                         tokenizer, remains_capital)

    def _load_data(self):
        data_fields = {
            'train': [['post', 'Sentence'], ['resp', 'Sentence'], ['score', Score]],
            'dev': [['post', 'Sentence'], ['resp', 'Sentence']],
            'test': [['post', 'Sentence'], ['resp', 'Sentence']],
        }
        return self._general_load_data(self._file_path, data_fields, \
                                       self._min_vocab_times, self._max_sent_length, None, self._invalid_vocab_times)

    def _general_load_data(self, file_path, data_fields, min_vocab_times, max_sent_length, max_turn_length,
                           invalid_vocab_times):
        r'''This function implements a general loading process.

        		Arguments:
        			file_path (str): A string indicating the path of dataset.
        			data_fields (dict, list, tuple): If it's a list(tuple), it must be a list of (key, field) pairs.
        				Field must be a DataField instance,
        				or a subclass of DataField(in this case, its instance will be used, assuming its constructor accepts no arguments),
        				or a string(in this case, the instance of the class, whose __name__ is field, will be used).

        				For example, data_fields=[['post', 'Sentence'], ['label', Label]] means that,
        				in the raw file, the first line is a sentence and the second line is a label. They are saved in a dict.
        				dataset = {'post': [line1, line3, line5, ...], 'label': [line2, line4, line6, ...]}

        				data_fields=[['key1', 'Session'], ['key2', Label()]], means that, in the raw file, the first *several lines*
        				is a session, *followed by an empty line*, and the next line is a label.
        				dataset = {'key1': [session1, session2, ...], 'key2': [label1, label2, ...]}

        				If it's a dict, different datasets may have different formats.(If `data_fields` is a list or a tuple, different datasets have the same format).
        				Its keys are the same as `self.key_name` that indicate the datasets, and the values are lists as mentioned above.
        				For example, data_fields = {'train': [['sess', 'Session'], ['label', 'Label']], 'test': [['sess', 'session']]},
        				means that the train set contains sessions and labels, but the test set only contains sessions.

        			min_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
        				not less than `min_vocab_times` in **training set** will be marked as valid words.
        			max_sent_length (int): All sentences longer than ``max_sent_length`` will be shortened
        				to first ``max_sent_length`` tokens.
        			max_turn_length (int): All sessions, whose turn length is longer than ``max_turn_length`` will be shorten to
        				first ``max_turn_length`` sentences. If the dataset don't contains sessions, this parameter will be ignored.
        			invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
        				not less than ``invalid_vocab_times`` in the **whole dataset** (except valid words) will be
        				marked as invalid words. Otherwise, they are unknown words, which are ignored both for
        				model or metrics.

        		Returns:
        			(tuple): containing:

        			* **all_vocab_list** (list): vocabulary list of the datasets,
        			  including valid and invalid vocabs.
        			* **valid_vocab_len** (int): the number of valid vocab.
        			  ``vocab_list[:valid_vocab_len]`` will be regarded as valid vocabs,
        			  while ``vocab_list[valid_vocab_len:]`` regarded as invalid vocabs.
        			* **data** (dict): a dict contains data.
        			* **data_size** (dict): a dict contains size of each item in data.
        		'''

        def get_fields(fields):
            assert isinstance(fields, list) or isinstance(fields, tuple)
            return [(data_key, DataField.get_field(field)) for data_key, field in fields]

        if isinstance(data_fields, dict):
            no_field_keys = [key for key in self.key_name if key not in data_fields]
            if no_field_keys:
                raise ValueError('There is no data fields for dataset(%s) ' % ', '.join(no_field_keys))
            try:
                data_fields = {key: get_fields(data_fields[key]) for key in self.key_name}
            except AssertionError:
                raise TypeError('If `data_field` is a dict, its value must be a list(or tuple) of lists(or tuples).')
        elif isinstance(data_fields, list) or isinstance(data_fields, tuple):
            data_fields = get_fields(data_fields)
            data_fields = {key: data_fields for key in self.key_name}
        else:
            raise TypeError('`data_fields` must be a dict, or a list, or a tuple.')

        # now data_fields is a dict. Keys are the same as self.key_name('train', 'test', 'dev', etc.). Each value is
        # a list(tuple) of lists(tuples), which means (data_key(str), data_field(DataField)) pairs.
        # For example,
        # data_fields == {'train': [['sent', Sentence()], ['label', Label()]],
        # 'test': [['sent', Sentence()], ['label', Label()]]}.
        # Note, different dataset may have different fields.

        special_tokens = set(self.ext_vocab)
        origin_data = {}
        for key in self.key_name:
            origin_data[key] = {data_key: [] for data_key, _ in data_fields[key]}
            with open("%s/%s.txt" % (file_path, key), encoding='utf-8') as f_file:
                while True:
                    try:
                        for data_key, field in data_fields[key]:
                            element = field.convert_to_tokens(field.get_next(f_file), self.tokenize)
                            for token in field.iter_tokens(element):
                                if token in special_tokens:
                                    raise RuntimeError(
                                        'The dataset contains special token "%s". This is not allowed.' % token)
                            origin_data[key][data_key].append(element)
                    except StopIteration:
                        break

        def chain_allvocab(dic, fields):
            vocabs = []
            for data_key, field in fields:
                for element in dic[data_key]:
                    vocabs.extend(field.iter_tokens(element))
            return vocabs

        raw_vocab_list = chain_allvocab(origin_data['train'], data_fields['train'])
        # Important: Sort the words preventing the index changes between
        # different runs
        vocab = sorted(Counter(raw_vocab_list).most_common(), \
                       key=lambda pair: (-pair[1], pair[0]))
        left_vocab = [x[0] for x in vocab if x[1] >= min_vocab_times]
        vocab_list = self.ext_vocab + list(left_vocab)
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        for key in self.key_name:
            if key == 'train':
                continue
            raw_vocab_list.extend(chain_allvocab(origin_data[key], data_fields[key]))

        vocab = sorted(Counter(raw_vocab_list).most_common(), \
                       key=lambda pair: (-pair[1], pair[0]))
        left_vocab = [x[0] for x in vocab if x[1] >= invalid_vocab_times and x[0] not in valid_vocab_set]
        vocab_list.extend(left_vocab)

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            for data_key, field in data_fields[key]:
                origin_data[key][data_key] = [field.convert_to_ids(element, word2id, self) for element in
                                              origin_data[key][data_key]]
                data[key][data_key] = [
                    field.cut(element, max_sent_length=max_sent_length, max_turn_length=max_turn_length) for element in
                    origin_data[key][data_key]]
                if key not in data_size:
                    data_size[key] = len(data[key][data_key])
                elif data_size[key] != len(data[key][data_key]):
                    raise RuntimeError(
                        "The data of input %s.txt contains different numbers of fields" % key)

            vocab = chain_allvocab(origin_data[key], data_fields[key])
            vocab_num = len(vocab)
            oov_num = sum([word not in word2id for word in vocab])
            invalid_num = sum([word not in valid_vocab_set for word in vocab]) - oov_num

            sent_length = []
            for data_key, field in data_fields[key]:
                sent_length.extend(
                    [len(sent) for element in origin_data[key][data_key] for sent in field.iter_sentence(element)])

            cut_word_num = np.sum(np.maximum(np.array(sent_length) - max_sent_length, 0))

            session_keys = [data_key for data_key, field in data_fields[key] if field.__class__ == Session]
            if session_keys:
                turn_length = list(
                    map(len, chain.from_iterable((origin_data[key][sess_key] for sess_key in session_keys))))
                max_turn_length_before_cut = max(turn_length)
                sent_num = sum(turn_length)
                cut_sentence_rate = np.sum(np.maximum(np.array(turn_length) - max_turn_length, 0)) / sent_num
            else:
                max_turn_length_before_cut = 1
                cut_sentence_rate = 0
            print(("%s set. invalid rate: %f, unknown rate: %f, max sentence length before cut: %d, " + \
                   "cut word rate: %f\n\tmax turn length before cut: %d, cut sentence rate: %f") % \
                  (key, invalid_num / vocab_num, oov_num / vocab_num, max(sent_length), \
                   cut_word_num / vocab_num, max_turn_length_before_cut, cut_sentence_rate))

        # calculate hash value
        hash_value = DataloaderHash(ignore_tokens=(self.go_id, self.eos_id, self.pad_id),
                                    unk_id=self.unk_id).hash_datasets(data, data_fields, vocab_list[len(
            self.ext_vocab):valid_vocab_len])
        self.__hash_value = hash_value

        return vocab_list, valid_vocab_len, data, data_size

    def get_batch(self, key, indexes):
        '''{LanguageProcessingBase.GET_BATCH_DOC_WITHOUT_RETURNS}

        Returns:
            (dict): A dict at least contains:

            * **post_length** (:class:`numpy.ndarray`): A 1-d array, the length of post in each batch.
              Size: ``[batch_size]``
            * **post** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form in posts.
              Only provide valid words. ``unk_id`` will be used if a word is not valid.
              Size: ``[batch_size, max(sent_length)]``
            * **post_allvocabs** (:class:`numpy.ndarray`): A 2-d padded array containing words of id
              form in posts. Provide both valid and invalid vocabs.
              Size: ``[batch_size, max(sent_length)]``
            * **resp_length** (:class:`numpy.ndarray`): A 1-d array, the length of response in each batch.
              Size: ``[batch_size]``
            * **resp** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form
              in responses. Only provide valid vocabs. ``unk_id`` will be used if a word is not valid.
              Size: ``[batch_size, max(sent_length)]``
            * **resp_allvocabs** (:class:`numpy.ndarray`):
              A 2-d padded array containing words of id form in responses.
              Provide both valid and invalid vocabs.
              Size: ``[batch_size, max(sent_length)]``

        Examples:
            >>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
            >>> #	"hello", "i", "am", "fine"]
            >>> # vocab_size = 9
            >>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
            >>> dataloader.get_batch('train', [0, 1])
            {
                "post_allvocabs": numpy.array([
                    [2, 5, 6, 10, 3],  # first post:  <go> are you fine <eos>
                    [2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
                ]),
                "post": numpy.array([
                    [2, 5, 6, 1, 3],   # first post:  <go> are you <unk> <eos>
                    [2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
                ]),
                "resp_allvocabs": numpy.array([
                    [2, 8, 9, 10, 3],  # first response:  <go> i am fine <eos>
                    [2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
                ]),
                "resp": numpy.array([
                    [2, 8, 1, 1, 3],   # first response:  <go> i <unk> <unk> <eos>
                    [2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
                ]),
                "post_length": numpy.array([5, 3]), # length of posts
                "resp_length": numpy.array([5, 3]), # length of responses
            }
        '''
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {}
        batch_size = len(indexes)
        res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)), dtype=int)
        res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)), dtype=int)
        res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        for i, j in enumerate(indexes):
            post = self.data[key]['post'][j]
            resp = self.data[key]['resp'][j]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id
        if key=='train':
            res['score']=np.array([self.data[key]['score'][i] for i in indexes])
        return res


def main():
    max_sent_length = 50
    loader = TranslationWithScore('./data/iwslt14_raml', 10, max_sent_length, 0, 'nltk', False)
    loader.restart("train",batch_size=2,shuffle=True)
    q=loader.get_next_batch("train")
    print(len(q['score']))
    print(q)

if __name__ == '__main__':
    main()
