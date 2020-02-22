# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Attentional Seq2seq using Scheduled sampling algorithm.

This code is basically the same as baseline_seq2seq_attn_main.py,
except using ScheduledEmbeddingTrainingHelper.

Scheduled Sampling Algorithm is described in https://arxiv.org/abs/1506.03099
"""

# pylint: disable=invalid-name, too-many-arguments, too-many-locals
import os
from io import open
import math
import importlib
import tensorflow as tf
import texar.tf as tx
from rouge import Rouge
import cotk
import numpy as np
import json


def set_seed(seed=1):
    np.random.seed(seed)
    tf.set_random_seed(seed)


set_seed()
flags = tf.flags


def inverse_sigmoid(i, FLAGS):
    return FLAGS.decay_factor / (
            FLAGS.decay_factor + math.exp(i / FLAGS.decay_factor))


def build_model(batch, loader, self_sampling_proba, config_model):
    """
    Assembles the seq2seq model.
    It is the same as build_model() in baseline_seq2seq_attn.py except
    using ScheduledEmbeddingTrainingHelper.
    """
    source_embedder = tx.modules.WordEmbedder(
        vocab_size=loader.vocab_size, hparams=config_model.embedder)

    encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.encoder)

    enc_outputs, _ = encoder(source_embedder(batch['source_text_ids']))

    target_embedder = tx.modules.WordEmbedder(
        vocab_size=loader.vocab_size, hparams=config_model.embedder)

    decoder = tx.modules.AttentionRNNDecoder(
        memory=tf.concat(enc_outputs, axis=2),
        memory_sequence_length=batch['source_length'],
        vocab_size=loader.vocab_size,
        hparams=config_model.decoder)

    helper = tx.modules.get_helper(
        helper_type='ScheduledEmbeddingTrainingHelper',
        inputs=target_embedder(batch['target_text_ids'][:, :-1]),
        sequence_length=batch['target_length'] - 1,
        embedding=target_embedder,
        sampling_probability=self_sampling_proba)

    training_outputs, _, _ = decoder(
        helper=helper, initial_state=decoder.zero_state(
            batch_size=tf.shape(batch['target_length'])[0], dtype=tf.float32))

    train_op = tx.core.get_train_op(
        tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=batch['target_text_ids'][:, 1:],
            logits=training_outputs.logits,
            sequence_length=batch['target_length'] - 1),
        hparams=config_model.opt)

    start_tokens = tf.ones_like(batch['target_length']) * \
                   loader.go_id
    beam_search_outputs, _, _ = \
        tx.modules.beam_search_decode(
            decoder_or_cell=decoder,
            embedding=target_embedder,
            start_tokens=start_tokens,
            end_token=loader.eos_id,
            beam_width=config_model.beam_width,
            max_decoding_length=60)

    return train_op, beam_search_outputs


def print_stdout_and_file(content, file):
    print(content)
    print(content, file=file)


def main(FLAGS=None):
    """Entrypoint.
    """

    if FLAGS is None:
        flags.DEFINE_string("config_model", "configs.config_model", "The model config.")
        flags.DEFINE_string("config_data", "configs.config_giga",
                            "The dataset config.")

        flags.DEFINE_float('decay_factor', 500.,
                           'The hyperparameter controling the speed of increasing '
                           'the probability of sampling from model')

        flags.DEFINE_string('output_dir', '.', 'where to keep training logs')
        flags.DEFINE_bool('cpu', False, 'whether to use cpu')
        flags.DEFINE_string('gpu', '0', 'use which gpu(s)')
        flags.DEFINE_bool('debug', False, 'if debug, skip the training process after one step')
        flags.DEFINE_bool('load', False, 'Whether to load existing checkpoint')
        flags.DEFINE_bool('infer', False, 'infer (use pretrained model)')

        FLAGS = flags.FLAGS

    if FLAGS.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    config_model = importlib.import_module(FLAGS.config_model)
    config_data = importlib.import_module(FLAGS.config_data)
    debug = FLAGS.debug
    load = FLAGS.load
    infer = FLAGS.infer
    dataset = FLAGS.config_data.split('.')[-1].split('_')[-1]
    print(f"dataset={dataset}")
    if not FLAGS.output_dir.endswith('/'):
        FLAGS.output_dir += '/'
    log_dir = FLAGS.output_dir + 'training_log_scheduled_sampling' + \
              '_decayf' + str(FLAGS.decay_factor) + '/' + dataset + '/'
    tx.utils.maybe_create_dir(log_dir)
    checkpoint_dir = './checkpoints/' + 'scheduled_sampling/' + dataset + '/'
    tx.utils.maybe_create_dir(checkpoint_dir)

    max_sent_length = 50
    loader = cotk.dataloader.SingleTurnDialog(f'./data/{dataset}', 10, max_sent_length, 0, 'nltk', False)
    batch_size = config_data.batch_size
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    batch = data_iterator.get_next()

    self_sampling_proba = tf.placeholder(shape=[], dtype=tf.float32)
    train_op, infer_outputs = \
        build_model(batch, loader, self_sampling_proba, config_model)

    def _train_epoch(sess, epoch_no, total_step_counter):
        data_iterator.switch_to_train_data(sess)
        training_log_file = \
            open(log_dir + 'training_log' + str(epoch_no) + '.txt', 'w',
                 encoding='utf-8')

        step = 0
        loader.restart("train", batch_size=batch_size, shuffle=True)
        batched_data = loader.get_next_batch("train")
        while batched_data is not None:
            sampling_proba_ = 1. - inverse_sigmoid(total_step_counter, FLAGS)
            loss = sess.run(train_op, feed_dict={
                self_sampling_proba: sampling_proba_,
                batch['source_text_ids']: batched_data['post'],
                batch['source_length']: batched_data['post_length'],
                batch['target_text_ids']: batched_data['resp'],
                batch['target_length']: batched_data['resp_length']
            })
            print("step={}, loss={:.4f}, self_proba={}".format(
                step, loss, sampling_proba_), file=training_log_file)
            if step % config_data.observe_steps == 0:
                print("step={}, loss={:.4f}, self_proba={}".format(
                    step, loss, sampling_proba_))
            training_log_file.flush()
            step += 1
            total_step_counter += 1
            batched_data = loader.get_next_batch("train")
            if debug:
                break

    # code below this line is exactly the same as baseline_seq2seq_attn_main.py

    def _eval_epoch(sess, mode, epoch_no):
        if mode == 'dev':
            data_iterator.switch_to_val_data(sess)
        else:
            data_iterator.switch_to_test_data(sess)
        loader.restart(mode, batch_size=batch_size)
        batched_data = loader.get_next_batch(mode)
        refs, hypos = [], []
        refs_id, hypos_id = [], []
        while batched_data is not None:
            fetches = [infer_outputs.predicted_ids[:, :, 0]]
            feed_dict = {
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                batch['source_text_ids']: batched_data['post'],
                batch['source_length']: batched_data['post_length'],
                batch['target_text_ids']: batched_data['resp'],
                batch['target_length']: batched_data['resp_length']
            }
            output_ids = sess.run(fetches, feed_dict=feed_dict)
            x = [loader.convert_ids_to_tokens(q, trim=True)[1:] for q in batched_data['resp']]
            target_texts = tx.utils.str_join(x)
            # print('x:{}\ntarget_texts:{}'.format(x, target_texts))
            y = [loader.convert_ids_to_tokens(q, trim=True) for q in output_ids[0]]
            output_texts = tx.utils.str_join(y)
            tx.utils.write_paired_text(
                target_texts, output_texts,
                log_dir + mode + '_results' + str(epoch_no) + '.txt',
                append=True, mode='h', sep=' ||| ')
            for hypo_id, ref_id in zip(output_ids[0], batched_data['resp']):
                if config_data.eval_metric == 'bleu':
                    hypos_id.append(hypo_id)
                    refs_id.append(ref_id)

            for hypo, ref in zip(output_texts, target_texts):
                if config_data.eval_metric == 'bleu':
                    hypos.append(hypo)
                    refs.append([ref])
                elif config_data.eval_metric == 'rouge':
                    hypos.append(tx.utils.compat_as_text(hypo))
                    refs.append(tx.utils.compat_as_text(ref))
            batched_data = loader.get_next_batch(mode)
            if debug:
                break

        if config_data.eval_metric == 'bleu':
            BleuMetric = cotk.metric.BleuCorpusMetric(loader)
            data = {'ref_allvocabs': refs_id, 'gen': hypos_id}
            BleuMetric.forward(data)
            result = BleuMetric.close()
            return result['bleu'], result
        elif config_data.eval_metric == 'rouge':
            rouge = Rouge()
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    def _calc_reward(score):
        """
        Return the bleu score or the sum of (Rouge-1, Rouge-2, Rouge-L).
        """
        if config_data.eval_metric == 'bleu':
            return score
        elif config_data.eval_metric == 'rouge':
            return sum([value['f'] for key, value in score.items()])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        if load and tf.train.latest_checkpoint(checkpoint_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        best_val_score = -1.
        total_step_counter = 1

        scores_file = open(log_dir + 'scores.txt', 'w', encoding='utf-8')
        if not infer:
            for i in range(config_data.num_epochs):
                _train_epoch(sess, i, total_step_counter)

                val_score, _ = _eval_epoch(sess, 'dev', i)
                test_score, result = _eval_epoch(sess, 'test', i)

                best_val_score = max(best_val_score, _calc_reward(val_score))
                if best_val_score == _calc_reward(val_score):
                    saver.save(sess, checkpoint_dir, global_step=i + 1)
                    with open(checkpoint_dir + 'result.json', 'w', encoding='utf-8') as file:
                        json.dump(result, file)
                if config_data.eval_metric == 'bleu':
                    print_stdout_and_file(
                        'val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                            i, val_score, best_val_score), file=scores_file)

                    print_stdout_and_file(
                        'test epoch={}, BLEU={:.4f}'.format(i, test_score),
                        file=scores_file)
                    print_stdout_and_file('=' * 50, file=scores_file)

                elif config_data.eval_metric == 'rouge':
                    print_stdout_and_file(
                        'valid epoch {}:'.format(i), file=scores_file)
                    for key, value in val_score.items():
                        print_stdout_and_file(
                            '{}: {}'.format(key, value), file=scores_file)
                    print_stdout_and_file('fsum: {}; best_val_fsum: {}'.format(
                        _calc_reward(val_score), best_val_score), file=scores_file)

                    print_stdout_and_file(
                        'test epoch {}:'.format(i), file=scores_file)
                    for key, value in test_score.items():
                        print_stdout_and_file(
                            '{}: {}'.format(key, value), file=scores_file)
                    print_stdout_and_file('=' * 110, file=scores_file)

                scores_file.flush()
        else:
            val_score, _ = _eval_epoch(sess, 'dev', 0)
            test_score, result = _eval_epoch(sess, 'test', 0)
            with open(log_dir + 'result.json', 'w', encoding='utf-8') as file:
                json.dump(result, file)


if __name__ == '__main__':
    main()
