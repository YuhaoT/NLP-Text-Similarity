import numpy as np
import  math
from gensim.models import Word2Vec


def create_word2vec_matrix(data_loader, min_count=20, size=100, window_size=3):
    """

    Arguments
    ---------
    `data_loader`: `list`-like
        A `list` (or `list` like object), each item of which is itself a list of tokens.
        For example:

                [
                    ['this', 'is', 'sentence', 'one'],
                    ['this', 'is', 'sentence', 'two.']
                ]

        Note that preprocesisng.DataLoader is exactly this kind of object.
    `min_count`: `int`
        The minimum count that a token has to have to be included in the vocabulary.
    `size`: `int`
        The dimensionality of the word vectors.
    `window_size`: `int`
        The window size. Read the assignment pdf if you don't know what that is.

    Returns
    -------
        `tuple(np.ndarray, dict)`:
            The first element will be an (V, `size`) matrix, where V is the
            resulting vocabulary size.

            The second element is a mapping from `int` to `str` of which word
            in the vocabulary corresponds to which index in the matrix.
    """

    word2vec_mat = None
    word2vec_id_to_tokens = None
    # NOTE : Your code should not be longer than ~2 lines, excepting the part
    # where build the int to string token mapping(and even that can be a one-liner).
    ####### Your code here ###############
    model = Word2Vec(sentences=data_loader, size = size, window = window_size, min_count = min_count)
    wordVec = model.wv
    word2vec_mat = np.zeros([len(model.wv.vocab), size])
    word2vec_id_to_tokens = {}
    for key, value in wordVec.vocab.items():
        word2vec_mat[value.index] = wordVec[key]
        word2vec_id_to_tokens[value.index] = key
    ####### End of your code #############

    return word2vec_mat, word2vec_id_to_tokens


def create_term_newsgroup_matrix(
    newsgroup_and_token_ids_per_post,
    id_to_tokens,
    id_to_newsgroups,
    tf_idf_weighing=False,
):
    """

    Arguments
    ---------
        newsgroup_and_token_ids_per_post: `list`
            Each item will be a `tuple` of length 2.
            Each `tuple` contains a `list` of token ids(`int`s), and a newsgroup id(`int`)
            Something like this:

                newsgroup_and_token_ids_per_post=[
                    ([0, 54, 3, 6, 7, 7], 0),
                    ([0, 4,  7], 0),
                    ([0, 463,  435, 656,  ], 1),
                ]

            The "newsgroup_and_token_ids_per_post" that main.read_processed_data() returns
            is exactly in this format.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.

                { 0: "hi", 1: "hello"  ...}

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.

                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        tf_idf_weighing: `bool`
            Whether to use TF IDF weighing in returned matrix.

    Returns
    -------
        `np.ndarray`:
            Shape will be (len(id_to_tokens), len(id_to_newsgroups)).
            That is it will be a VxD matrix where V is vocabulary size and D is number of newsgroups.

            Note, you may choose to remove the row corresponding to the "UNK" (which stands for unknown)
            token.
    """

    V = len(id_to_tokens)
    D = len(id_to_newsgroups)

    mat = None
    ####### Your code here ###############
    # skip tf_idf for now
    mat = []
    UNK_id = None
    for i in range(V):
        mat.append([])

    for key, value in id_to_tokens.items():
        if value == 'UNK':
            UNK_id = key

    for pair in newsgroup_and_token_ids_per_post:
        d = pair[1]
        for w in pair[0]:
            temp = [0] * D
            if not mat[w]:
                temp[d] += 1
                mat[w] = temp
            else:
                mat[w][d] += 1

    del mat[UNK_id]
    mat = np.array(mat)

    if tf_idf_weighing:
        #TODO: finish tf_idf
        # idf
        # Return the natural logarithm of 1+x (base e). The result is calculated in a way which is accurate for x near zero.
        idfs = ([math.log(D/np.count_nonzero(row)) for row in mat])
        # tf
        tfs = [(1 + math.log10(c)) if c > 0 else 0 for row in mat for c in row]
        tfs = np.array(tfs)
        tfs.shape = (V-1, D)

        for r_id in range(V-1):
            for col_id in range(D):
                tf_idf = tfs[r_id][col_id] * idfs[r_id]
                tfs[r_id][col_id] = tf_idf
        mat = tfs

    ####### End of your code #############

    return mat


def create_term_context_matrix(
    newsgroup_and_token_ids_per_post,
    id_to_tokens,
    id_to_newsgroups,
    ppmi_weighing=False,
    window_size=5,
):
    """

    Arguments
    ---------
        newsgroup_and_token_ids_per_post: `list`
            Each item will be a `tuple` of length 2.
            Each `tuple` is a post, contains a `list` of token ids(`int`s), and a newsgroup id(`int`)
            Something like this:

                newsgroup_and_token_ids_per_post=[
                    ([0, 54, 3, 6, 7, 7], 0),
                    ([0, 4,  7], 0),
                    ([0, 463,  435, 656,  ], 1),
                ]

            The "newsgroup_and_token_ids_per_post" that main.read_processed_data() returns
            is exactly in this format.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.

                { 0: "hi", 1: "hello  ...}

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.

                { 0: "hi", 1: "hello  ...}
                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        ppmi_weighing: `bool`
            Whether to use PPMI weighing in returned matrix.

    Returns
    -------
        `np.ndarray`:
            Shape will be (len(id_to_tokens), len(id_to_tokens)).
            That is it will be a VxV matrix where V is vocabulary size.

            Note, you may choose to remove the row/column corresponding to the "UNK" (which stands for unknown)
            token.
    """

    V = len(id_to_tokens)
    mat = None
    ####### Your code here ###############
    mat = []
    UNK_id = None
    for i in range(V):
        mat.append([])

    for key, value in id_to_tokens.items():
        if value == 'UNK':
            UNK_id = key

    for pair in newsgroup_and_token_ids_per_post:
        length = len(pair[0])
        for i in range(length):
            index = pair[0][i]
            c = window_size
            while c > 0:
                l = i - c
                r = i + c
                if l < 0:
                    l = None
                if r > (length-1):
                    r = None

                if not mat[index]:
                    temp = [0] * V
                    if l != None:
                        temp[pair[0][l]] += 1
                    if r != None:
                        temp[pair[0][r]] += 1
                    mat[index] = temp
                else:
                    if l != None:
                        mat[index][pair[0][l]] += 1
                    if r != None:
                        mat[index][pair[0][r]] += 1
                c -= 1

    # delete UNK from matrix
    del mat[UNK_id]
    mat = np.array(mat)
    mat_col = mat.T
    mat_col = mat_col.tolist()
    del mat_col[UNK_id]
    mat_col = np.array(mat_col)
    mat = mat_col.T

    # TODO: finish ppmi
    if ppmi_weighing:
        mat = mat + 1  # add 1 smoothing
        words_total = np.count_nonzero(mat)
        Pwd = [x/words_total for row in mat for x in row] #1D array
        Pwd = np.array(Pwd)
        Pwd.shape = (V-1, V-1)
        Pw = [sum(row)/words_total for row in mat]

        mat_col = mat.T
        Pd = [sum(row)/words_total for row in mat_col]

        pmi = []
        for r_id in range(V - 1):
            for w_id in range(V - 1):
                p =Pwd[r_id][w_id]/(Pw[r_id] * Pd[w_id]) # 1D array
                if p == 0:
                    pmi.append(p)
                else:
                    pmi.append(math.log2(p))

        ppmi = [x if x > 0 else 0 for x in pmi]
        ppmi = np.array(ppmi)
        ppmi.shape = (V-1, V-1)
        mat = ppmi

    ####### End of your code #############
    return mat


def compute_cosine_similarity(a, B):
    """Cosine similarity.

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The cosine similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    cos_N = []
    for b in B:
        cos_sim = (a @ b)/(np.linalg.norm(a) * np.linalg.norm(b))
        cos_N.append(cos_sim)

    N = np.array(cos_N)
    return N
    ####### End of your code #############


def compute_jaccard_similarity(a, B):
    """

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The Jaccard similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    jac = []
    for b in B:
        min_sum = sum([min(x, y) for x, y in zip(a, b)])
        max_sum = sum([max(x, y) for x, y in zip(a, b)])
        j = min_sum/max_sum
        jac.append(j)
    return np.array(jac)
    ####### End of your code #############


def compute_dice_similarity(a, B):
    """

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The Dice similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    dice = []
    for b in B:
        min_sum = sum([min(x, y) for x, y in zip(a, b)])
        denominator = sum([x+y for x, y in zip(a, b)])
        d = 2*(min_sum)/(denominator)
        dice.append(d)

    return np.array(dice)
    ####### End of your code #############
