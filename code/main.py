import numpy as np

import vec_spaces
from preprocessing import read_processed_data, DataLoader

#self import
import random

def test_newsgroup_similarity(
    mat, id_to_newsgroups, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, D)
            Each column is "newsgroup" vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.

                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############
    docs = mat.T
    doc_sim = []
    for d in docs:
        sim = sim_func(d, docs)
        i = np.argmax(sim)
        doc_name = id_to_newsgroups[i]
        doc_sim.append(doc_name)

    return print(doc_sim)

    ####### End of your code #############


def test_word_similarity(
    mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)
            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.

                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############
    test_words = ['man', 'boston', 'save', '226', 'doctrine', 'he', 'what', 'about', 'other', 'sounds']
    tokens_to_ids = {token: id for id, token in id_to_tokens.items()}
    chosen_token_ids = [tokens_to_ids[token] for token in test_words]

    similarity_words = []
    test_words_vec = mat[chosen_token_ids]

    for w in test_words_vec:
        sims = sim_func(w, mat)
        sorted = np.sort(sims)
        similarity_word = sorted[-2]
        i = np.where(sims == similarity_word)

        similarity_words.append(id_to_tokens[i[0][0]])

    return print(similarity_words)
    ####### End of your code #############


def test_word2vec_similarity(
    mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)

            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.

                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """
    ####### Your code here ###############
    pass
    ####### End of your code #############


def main():
    ####### Your code here ###############
    data = read_processed_data()

    mat_1= vec_spaces.create_term_newsgroup_matrix(data['newsgroup_and_token_ids_per_post'], data['id_to_tokens'], data['id_to_newsgroups'])

    test_newsgroup_similarity(mat_1, data['id_to_newsgroups'])
    test_word_similarity(mat_1, data['id_to_tokens'])
    test_word_similarity(mat_1, data['id_to_tokens'], sim_func = vec_spaces.compute_jaccard_similarity)
    test_word_similarity(mat_1, data['id_to_tokens'], sim_func = vec_spaces.compute_dice_similarity)

    #part 2

    mat_2 = vec_spaces.create_term_context_matrix(data['newsgroup_and_token_ids_per_post'], data['id_to_tokens'], data['id_to_newsgroups'])

    test_word_similarity(mat_2, data['id_to_tokens'])
    test_word_similarity(mat_2, data['id_to_tokens'], sim_func = vec_spaces.compute_jaccard_similarity)
    test_word_similarity(mat_2, data['id_to_tokens'], sim_func = vec_spaces.compute_dice_similarity)

    #part 3
    d = DataLoader(lower_case = True, include_newsgroup=False)
    data = vec_spaces.create_word2vec_matrix(d) # vocab = 10537
    test_word_similarity(data[0], data[1])
    test_word_similarity(data[0], data[1], sim_func = vec_spaces.compute_jaccard_similarity)
    test_word_similarity(data[0], data[1], sim_func = vec_spaces.compute_dice_similarity)

    #part 5

    ####### End of your code #############
    mat_ppmi = vec_spaces.create_term_context_matrix(data['newsgroup_and_token_ids_per_post'], data['id_to_tokens'],
                                                  data['id_to_newsgroups'], ppmi_weighing = True)

    test_word_similarity(mat_ppmi, data['id_to_tokens'])
    test_word_similarity(mat_ppmi, data['id_to_tokens'], sim_func=vec_spaces.compute_jaccard_similarity)
    test_word_similarity(mat_ppmi, data['id_to_tokens'], sim_func=vec_spaces.compute_dice_similarity)

    mat_tf_idf = vec_spaces.create_term_newsgroup_matrix(data['newsgroup_and_token_ids_per_post'], data['id_to_tokens'],
                                                    data['id_to_newsgroups'], tf_idf_weighing = True)

    test_newsgroup_similarity(mat_tf_idf, data['id_to_newsgroups'])
    test_word_similarity(mat_tf_idf, data['id_to_tokens'])
    test_word_similarity(mat_tf_idf, data['id_to_tokens'], sim_func=vec_spaces.compute_jaccard_similarity)
    test_word_similarity(mat_tf_idf, data['id_to_tokens'], sim_func=vec_spaces.compute_dice_similarity)

if __name__ == "__main__":
    main()
