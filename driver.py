import argparse

from pyhtmm.htmm import EM, HTMM
from pyhtmm.process import read_train_documents, process_doc
from pyhtmm.utils import config_logger, save_pickle, load_pickle


if __name__ == "__main__":
    word_index_filepath = './data/pickle/word_index.pickle'
    index_word_filepath = './data/pickle/index_word.pickle'
    model_filepath = './data/pickle/model.pickle'
    model_trained_filepath = './data/pickle/trained_model.pickle'
    htmm_model_trained_filepath = './data/pickle/htmm_trained_model.pickle'
    docs_path = './data/pickle/docs.pickle'

    # config_logger()

    parser = argparse.ArgumentParser(description='PyHTMM interface')
    parser.add_argument('--predict', type=str,
                        help='sentence to predict topic')
    parser.add_argument('--topwords', type=int,
                        help='number of top words to print for each topics')
    parser.add_argument('--iters', type=int, default=10,
                        help='number of iterations to train')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of workers to train')
    parser.add_argument('-train',action='store_true')
    parser.add_argument('-process',action='store_true')

    args = parser.parse_args()

    if args.predict != None:
        try:
            htmm_model = load_pickle(htmm_model_trained_filepath)
            word_index = load_pickle(word_index_filepath)
        except:
            print("model does not exist at %s" % (htmm_model_trained_filepath))
            print("word_index does not exist at %s" % (word_index_filepath))
            exit()
        print(htmm_model.predict_topic(process_doc(args.predict, word_index)))
        exit()

    if args.process:
        print("start processing...")
        docs, word_index, index_word = read_train_documents('./data/laptops/')
        save_pickle(word_index, word_index_filepath)
        save_pickle(index_word, index_word_filepath)
        save_pickle(docs, docs_path)
    else:
        try:
            word_index = load_pickle(word_index_filepath)
            index_word = load_pickle(index_word_filepath)
            docs = load_pickle(docs_path)
        except:
            print("docs is not processed at %s, try add -process flag" % (docs_path))
            exit()

    if not args.train:
        try:
            model = load_pickle(model_trained_filepath)
        except:
            print("model does not exist at %s, try add -train flag" % (model_trained_filepath))
            exit()
    else:
        # train model
        print("start training...")
        model = EM(docs, len(word_index), num_workers=args.workers)
        model.save(model_filepath)

        model.load_prior('./data/laptops_bootstrapping_test.dat', word_index)
        model.infer(iters=args.iters)

        model.save(model_trained_filepath)
        model.save_HTMM_model(htmm_model_trained_filepath)

    if args.topwords != None:
        model.print_top_word(index_word, args.topwords)
