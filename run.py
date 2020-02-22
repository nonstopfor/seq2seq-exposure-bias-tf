# coding:utf-8

def run():
    import importlib
    from tensorflow import flags
    flags.DEFINE_string("config_model", "configs.config_model", "The model config.")
    flags.DEFINE_string("config_data", "configs.config_giga",
                        "The dataset config.")
    flags.DEFINE_float('decay_factor', 500.,
                       'The hyperparameter controling the speed of increasing '
                       'the probability of sampling from model')
    flags.DEFINE_integer('n_samples', 10,
                         'number of samples for every target sentence')
    flags.DEFINE_float('tau', 0.4, 'the temperature in RAML algorithm')

    flags.DEFINE_string('output_dir', '.', 'where to keep training logs')
    flags.DEFINE_bool('cpu', False, 'whether to use cpu')
    flags.DEFINE_string('gpu', '0', 'use which gpu(s)')
    flags.DEFINE_bool('debug', False, 'if debug, skip the training process after one step')
    flags.DEFINE_bool('load', False, 'Whether to load existing checkpoint')
    flags.DEFINE_string('script', 'python_test', 'which script to use')
    flags.DEFINE_bool('infer',False,'infer (use pretrained model)')
    FLAGS = flags.FLAGS
    # print(FLAGS.load)
    module_name = FLAGS.script
    # from python_test import main
    module = importlib.import_module(module_name)
    module.main(FLAGS)
    # from main import main
    # main(args)


if __name__ == '__main__':

    run()
