import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # read text into python, make everything lowercase, remove punctuation (~10s)
    def getText():
        valid = 'abcdefghijklmnopqrstuvwxyz '
        with open("AllCombined.txt") as f:
            raw = f.read()
        low = raw.lower().replace("\n", " ")
        letters_only = ''.join(char for char in low if char in valid)
        return letters_only

    txt = getText()[:10000000]
    print(txt[:500])
    return (txt,)


@app.cell
def _(txt):
    # tokenize and build vocab
    from collections import Counter
    vocab_sz = 2000
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    # stop_words = []
    lst = [word for word in txt.split() if word != '']
    vocab = [word for word, cnt in Counter(lst).most_common(vocab_sz) if word not in stop_words]
    vocab_sz = len(vocab)

    vocab_st = set(vocab)
    dict = {word: i for i, word in enumerate(vocab)}
    data = [dict[word] for word in lst if word in vocab_st]
    print(data[:20])
    print(len(data))
    print(len(vocab))
    print(len(stop_words))
    return data, dict, vocab, vocab_sz


@app.cell
def _(data):
    # training data, loop over the text to generate target and context
    import numpy as np
    import Neural as Neur
    x = []
    y = []
    window = 3
    for i in range(window, len(data) - window):
        start = i - window
        end = i + window + 1
        context = []
        for j in range(start, end):
            if i != j:
                context.append(data[j])
        x.append(context)
        y.append(data[i])

    Y_train = np.array(y)
    X_train = np.array(x)
    print(len(Y_train))
    return Neur, X_train, Y_train, np, window


@app.cell
def _(Neur, X_train, Y_train, np, vocab_sz, window):
    # neural network!
    import importlib
    from tqdm import tqdm
    importlib.reload(Neur)

    sz = len(Y_train)

    def get_cbow(context):
        mat = np.zeros((len(context), vocab_sz))
        for idx, row in enumerate(context):
            for word in row:
                mat[idx, word] += 1.0 / (2 * window)
        return mat

    def get_one_hot(indices):
        mat = np.zeros((len(indices), vocab_sz))
        mat[np.arange(len(indices)), indices] = 1
        return mat

    def build_net_sgd(dim, lr, epochs, decay, batch=128):
        history = []
        history_00 = []

        N = Neur.Network([Neur.Dense(vocab_sz, dim), Neur.Dense(dim, vocab_sz)])
        loss_layer = Neur.Ilogit_and_KL(None)
        N_a = Neur.Network([N, loss_layer])

        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(sz)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            batch_loop = tqdm(range(0, len(X_shuffled), batch),
                              desc=f"Epoch {epoch + 1}/{epochs}",
                              unit="batch")

            batches = 0
            for i in batch_loop:

                X_cur = get_cbow(X_shuffled[i:i+batch])
                Y_cur = get_one_hot(Y_shuffled[i:i+batch])

                loss_layer.save_D = Y_cur.T
                loss = N_a.forward(X_cur.T)
                total_loss += loss
                grads, in_grad = N_a.backward(None)
                history_00.append(grads[0])
                params = N_a.get_params()
                new_params = params - (lr * grads)

                N_a.set_params(new_params)

                if i * batch % 101 == 0:
                    batch_loop.set_postfix({"loss": f"{loss:.2f}"})

                batches += 1

            avg_loss = total_loss / batches
            history.append(avg_loss)
            # print(f"=> Average Loss: {avg_loss:.4f}")

            lr *= decay

        return N, history, history_00

    return build_net_sgd, get_cbow, get_one_hot, sz, tqdm


@app.cell
def _(Neur, X_train, Y_train, get_cbow, get_one_hot, np, sz, tqdm, vocab_sz):
    def build_net_red(dim, epochs, batch=128):
        history = []
        history_00 = []

        N = Neur.Network([Neur.Dense(vocab_sz, dim), Neur.Dense(dim, vocab_sz)])
        loss_layer = Neur.Ilogit_and_KL(None)
        N_a = Neur.Network([N, loss_layer])

        beta3, ell, eps = 0.9, 1.0, 1e-4
        chat = 0.0
        k = 0

        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(sz)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            batch_loop = tqdm(range(0, len(X_shuffled), batch),
                              desc=f"Epoch {epoch + 1}/{epochs}",
                              unit="batch")
            batches = 0

            for i in batch_loop:

                X_cur = get_cbow(X_shuffled[i:i+batch])
                Y_cur = get_one_hot(Y_shuffled[i:i+batch])

                loss_layer.save_D = Y_cur.T
                lv = N_a.forward(X_cur.T)
                total_loss += lv
                g, in_grad = N_a.backward(None)
                history_00.append(g[0])

                d = g
                nrm2 = np.dot(d, d)
                p = N_a.get_params()

                # probe loss at p +/- eps*d to get directional 2nd diff
                N_a.set_params(p + eps * d)
                lp = N_a.forward(X_cur.T)
                N_a.set_params(p - eps * d)
                lm = N_a.forward(X_cur.T)
                N_a.set_params(p)

                ck = max(abs((lp + lm - 2 * lv) / (eps ** 2)) / nrm2, 1e-20)
                chat = beta3 * chat + (1 - beta3) * ck
                ctilde = chat / (1 - beta3 ** (k + 1))
                Lk = max(ctilde, ck)
                rk = np.dot(d, g) / (2 * nrm2 * Lk)
                N_a.set_params(p - ell * rk * d)

                if i * batch % 101 == 0:
                    batch_loop.set_postfix({"loss": f"{lv:.2f}"})

                k += 1
                batches += 1

            avg_loss = total_loss / batches
            history.append(avg_loss)

        return N, history, history_00

    return (build_net_red,)


@app.cell
def _(Neur, X_train, Y_train, get_cbow, get_one_hot, np, sz, tqdm, vocab_sz):
    def build_net_adam(dim, lr, epochs, batch=128):
        history = []
        history_00 = []

        N = Neur.Network([Neur.Dense(vocab_sz, dim), Neur.Dense(dim, vocab_sz)])
        loss_layer = Neur.Ilogit_and_KL(None)
        N_a = Neur.Network([N, loss_layer])
        b1, b2, e = 0.9, 0.999, 1e-8

        m = np.zeros_like(N_a.nb_params)
        v = np.zeros_like(N_a.nb_params)
        k = 0

        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(sz)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            batch_loop = tqdm(range(0, len(X_shuffled), batch),
                              desc=f"Epoch {epoch + 1}/{epochs}",
                              unit="batch")
            batches = 0

            for i in batch_loop:

                X_cur = get_cbow(X_shuffled[i:i+batch])
                Y_cur = get_one_hot(Y_shuffled[i:i+batch])

                loss_layer.save_D = Y_cur.T
                lv = N_a.forward(X_cur.T)
                total_loss += lv
                g, in_grad = N_a.backward(None)
                history_00.append(g[0])

                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g**2
                mh = m / (1 - b1 ** (k + 1))
                vh = v / (1 - b2 ** (k + 1))

                N_a.set_params(N_a.get_params() - lr * mh / (np.sqrt(vh) + e))

                if i * batch % 101 == 0:
                    batch_loop.set_postfix({"loss": f"{lv:.2f}"})

                k += 1
                batches += 1

            avg_loss = total_loss / batches
            history.append(avg_loss)

        return N, history, history_00

    return (build_net_adam,)


@app.cell
def _(build_net_adam, build_net_red, build_net_sgd):
    # Run and plot
    import matplotlib.pyplot as plt
    import itertools

    """
    dims_test = [50]
    lr_test = [0.1]
    epochs = [10]
    decays = [1.0]
    batches = [sz,256,8]
    """

    N_list = []
    hist_list = []
    hist2_list = []

    N, history, history2 = build_net_adam(50, 0.005, 10, 64)
    N_list.append(N)
    hist_list.append(history)
    hist2_list.append(history2)

    N, history, history2 = build_net_red(50, 10, 64)
    N_list.append(N)
    hist_list.append(history)
    hist2_list.append(history2)

    N, history, history2 = build_net_sgd(50, 0.05, 10, 1.0, 64)
    N_list.append(N)
    hist_list.append(history)
    hist2_list.append(history2)

    """
    for dim, lr, epoch, decay, batch in itertools.product(dims_test, lr_test, epochs, decays, batches):

        print(f"Dimensions: {dim}, Learning Rate: {lr}, Epochs: {epoch}, Decay: {decay}, Batch Size: {batch}")
        N, history, history2 = build_net(dim, lr, epoch, decay, batch)

        plt.plot(range(1, epoch+1), history, marker='o', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

        plt.plot(history2, marker='', color='b')
        plt.xlabel('Iteration')
        plt.ylabel('W[0][0]')
        plt.grid(True)
        plt.show()

        N_list.append(N)
        hist_list.append(history)
        hist2_list.append(history2)
    """
    return N_list, hist2_list, hist_list, plt


@app.cell
def _(hist2_list, np, plt):
    data_adam = hist2_list[0]
    data_red = hist2_list[1]
    data_sgd = hist2_list[2]

    x_adam = np.linspace(0, 10, len(data_adam))
    x_red = np.linspace(0, 10, len(data_red))
    x_sgd = np.linspace(0, 10, len(data_sgd))

    plt.plot(x_adam, data_adam, color='g', alpha=0.3, label='Adam')
    plt.plot(x_red, data_red, color='r', alpha=0.5, label='RED')
    plt.plot(x_sgd, data_sgd, color='b', alpha=0.2, label='SGD')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient of W[0][0]')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(hist_list, plt):
    # plt.plot(range(1, 10+1), hist_list[0], marker='o', color='k', label='GD')
    plt.plot(range(1, 10+1), hist_list[0], marker='o', color='r', label='Adam')
    plt.plot(range(1, 10+1), hist_list[1], marker='o', color='g', label='RED')
    plt.plot(range(1, 10+1), hist_list[2], marker='o', color='b', label='SGD')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(N_list, dict, np, vocab):
    embed = N_list[2].list_layers[0].A.T
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    norm_embed = embed / norms

    def get_similar(word, top_k=5):
        if word not in dict:
            print(f"'{word}' not found.")
            return

        word_idx = dict[word]
        word_vec = norm_embed[word_idx]

        similar = norm_embed @ word_vec
        nearest = np.argsort(similar)[::-1]

        print(f"Words most similar to '{word}':")
        for i in range(1, top_k + 1):
            idx = nearest[i]
            score = similar[idx]
            print(f"  - {vocab[idx]}: {score:.4f}")

    get_similar("king")
    get_similar("apple")
    get_similar("war")
    get_similar("queen")
    get_similar("battle")
    return (norm_embed,)


@app.cell
def _(dict, norm_embed, np, vocab):
    def find_analogy(a, b, c, top_k=1):
        if any(w not in dict for w in [a, b, c]):
            print("One of the words is missing from the vocab.")
            return

        vec_a = norm_embed[dict[a]]
        vec_b = norm_embed[dict[b]]
        vec_c = norm_embed[dict[c]]

        target = vec_b - vec_a + vec_c
        target = target / np.linalg.norm(target)
        similar = norm_embed @ target
        nearest = np.argsort(similar)[::-1]

        print(f"'{a}' is to '{b}' as '{c}' is to:")

        count = 0
        for idx in nearest:
            word = vocab[idx]

            if word not in [a, b, c]:
                score = similar[idx]
                print(f"  {count + 1}. {word} ({score:.4f})")
                count += 1

            if count == top_k:
                break

    find_analogy("man", "king", "woman", top_k=3) # queen
    find_analogy("son", "father", "daughter", top_k=3) # mother
    find_analogy("boy", "brother", "girl", top_k=3) # sister
    find_analogy("paris", "france", "berlin", top_k=3) # germany
    find_analogy("france", "paris", "japan", top_k=3) # tokyo
    find_analogy("bright", "dark", "white", top_k=3) # black
    find_analogy("small", "smaller", "big", top_k=3) # bigger
    find_analogy("writing", "wrote", "making", top_k=3) # made
    find_analogy("america", "american", "canada", top_k=3) # canadian
    return
