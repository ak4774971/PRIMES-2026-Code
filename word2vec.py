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

    txt = getText()[:5000000]
    print(txt[:500])
    return (txt,)


@app.cell
def _(txt):
    # tokenize and build vocab
    from collections import Counter
    vocab_sz = 3000
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    lst = [word for word in txt.split() if word != '']
    filtered = [word for word in lst if word not in stop_words]
    vocab = [word for word, cnt in Counter(filtered).most_common(vocab_sz) if word not in stop_words]
    vocab_sz = len(vocab)

    vocab_st = set(vocab)
    dict = {word: i for i, word in enumerate(vocab)}
    data = [dict[word] for word in filtered if word in vocab_st]
    print(data[:20])
    print(len(data))
    print(len(vocab))
    print(len(stop_words))
    return data, dict, vocab, vocab_st, vocab_sz


@app.cell
def _(data):
    # training data, loop over the text to generate target and context
    import numpy as np
    import Neural as Neur
    np.random.seed(1000)
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


    x = np.array(x)
    y = np.array(y)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    end = int(len(x_shuffled) * 0.9)

    X_train = x_shuffled[:end]
    Y_train = y_shuffled[:end]
    X_test = x_shuffled[end:]
    Y_test = y_shuffled[end:]
    print(len(Y_train))
    return Neur, X_test, X_train, Y_test, Y_train, np, window


@app.cell
def _(Neur, Y_train, np, vocab_sz, window):
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

    return get_cbow, get_one_hot, sz, tqdm


@app.cell
def _(Neur, X_train, Y_train, get_cbow, get_one_hot, np, sz, tqdm, vocab_sz):
    import time
    import copy

    def step_sgd(N_a, state, g):
        params = N_a.get_params()
        N_a.set_params(params - state['lr'] * g)
        return state

    def step_red(N_a, state, g, lv, X):
        beta3, ell, eps = 0.9, 1.0, 1e-4

        d = g
        nrm2 = np.dot(d, d)
        p = N_a.get_params()

        N_a.set_params(p + eps * d)
        lp = N_a.forward(X.T)
        N_a.set_params(p - eps * d)
        lm = N_a.forward(X.T)
        N_a.set_params(p)

        ck = max(abs((lp + lm - 2 * lv) / (eps ** 2)) / nrm2, 1e-20)
        state['chat'] = beta3 * state['chat'] + (1 - beta3) * ck
        ctilde = state['chat'] / (1 - beta3 ** (state['k'] + 1))
        Lk = max(ctilde, ck)
        rk = np.dot(d, g) / (2 * nrm2 * Lk)

        N_a.set_params(p - ell * rk * d)
        state['k'] += 1
        return state

    def step_adam(N_a, state, g):
        b1, b2, e = 0.9, 0.999, 1e-8

        state['m'] = b1 * state['m'] + (1 - b1) * g
        state['v'] = b2 * state['v'] + (1 - b2) * g**2
        mh = state['m'] / (1 - b1 ** (state['k'] + 1))
        vh = state['v'] / (1 - b2 ** (state['k'] + 1))

        p = N_a.get_params()
        p -= state['lr'] * mh / (np.sqrt(vh) + e)
        N_a.set_params(p)
        state['k'] += 1
        return state

    def step_adahessian(N_a, state, g, loss_layer, X, Y):
        b1, b2, e, eps = 0.9, 0.999, 1e-4, 1e-4
        p = N_a.get_params()
        z = np.random.choice([-1.0, 1.0], size=len(p))
        N_a.set_params(p + eps * z)
        loss_layer.save_D = Y.T
        N_a.forward(X.T)
        gp, _ = N_a.backward(None)

        N_a.set_params(p)
        D = z * (gp - g) / eps
        D = np.clip(D, -2.0, 2.0)

        state['m'] = b1 * state['m'] + (1 - b1) * g
        state['v'] = b2 * state['v'] + (1 - b2) * (D**2)
        mh = state['m'] / (1 - b1 ** (state['k'] + 1))
        vh = state['v'] / (1 - b2 ** (state['k'] + 1))

        p -= state['lr'] * mh / (np.sqrt(vh) + e)
        N_a.set_params(p)
        state['k'] += 1
        return state

    def step_redm(N_a, state, g, lv, X):
        beta3, b1, ell, eps = 0.9, 0.9, 1.0, 1e-4
        state['m'] = b1 * state['m'] + (1 - b1) * g
        mh = state['m'] / (1 - b1 ** (state['k'] + 1))
        d = mh
        nrm2 = np.dot(d, d)
        p = N_a.get_params()

        N_a.set_params(p + eps * d)
        lp = N_a.forward(X.T)
        N_a.set_params(p - eps * d)
        lm = N_a.forward(X.T)
        N_a.set_params(p)

        ck = max(abs((lp + lm - 2 * lv) / (eps ** 2)) / nrm2, 1e-20)
        state['chat'] = beta3 * state['chat'] + (1 - beta3) * ck
        ctilde = state['chat'] / (1 - beta3 ** (state['k'] + 1))
        Lk = max(ctilde, ck)
        rk = 1 / (2 * Lk)

        p -= ell * rk * d
        N_a.set_params(p)
        state['k'] += 1
        return state

    def step_spadam(N_a, state, g):
        b1, b2, e = 0.9, 0.999, 1e-8

        active = np.where(g != 0.0)[0]
        state['m'][active] = b1 * state['m'][active] + (1 - b1) * g[active]
        state['v'][active] = b2 * state['v'][active] + (1 - b2) * g[active]**2
        mh = state['m'][active] / (1 - b1 ** (state['k'] + 1))
        vh = state['v'][active] / (1 - b2 ** (state['k'] + 1))

        p = N_a.get_params()
        p[active] -= state['lr'] * mh / (np.sqrt(vh) + e)
        N_a.set_params(p)
        state['k'] += 1
        return state

    def step_spadahessian(N_a, state, g, loss_layer, X, Y):
        b1, b2, e, eps = 0.9, 0.999, 1e-4, 1e-4

        active = np.where(g != 0.0)[0]
        p = N_a.get_params()
        z = np.zeros_like(p)
        z[active] = np.random.choice([-1.0, 1.0], size=len(active))
        N_a.set_params(p + eps * z)
        loss_layer.save_D = Y.T
        N_a.forward(X.T)
        gp, _ = N_a.backward(None)

        N_a.set_params(p)
        D = z * (gp - g) / eps
        D = np.clip(D, -2.0, 2.0)

        state['m'][active] = b1 * state['m'][active] + (1 - b1) * g[active]
        state['v'][active] = b2 * state['v'][active] + (1 - b2) * (D[active]**2)
        mh = state['m'][active] / (1 - b1 ** (state['k'] + 1))
        vh = state['v'][active] / (1 - b2 ** (state['k'] + 1))

        p[active] -= state['lr'] * mh / (np.sqrt(vh) + e)
        N_a.set_params(p)
        state['k'] += 1
        return state

    def step_spredm(N_a, state, g, lv, X):
        beta3, b1, ell, eps = 0.9, 0.9, 1.0, 1e-4

        active = np.where(g != 0.0)[0]
        state['m'][active] = b1 * state['m'][active] + (1 - b1) * g[active]
        mh = state['m'][active] / (1 - b1 ** (state['k'] + 1))
        d = np.zeros_like(g)
        d[active] = mh
        nrm2 = np.dot(d, d)
        p = N_a.get_params()

        N_a.set_params(p + eps * d)
        lp = N_a.forward(X.T)
        N_a.set_params(p - eps * d)
        lm = N_a.forward(X.T)
        N_a.set_params(p)

        ck = max(abs((lp + lm - 2 * lv) / (eps ** 2)) / nrm2, 1e-20)
        state['chat'] = beta3 * state['chat'] + (1 - beta3) * ck
        ctilde = state['chat'] / (1 - beta3 ** (state['k'] + 1))
        Lk = max(ctilde, ck)
        rk = 1 / (2 * Lk)

        p[active] -= ell * rk * d[active]
        N_a.set_params(p)
        state['k'] += 1
        return state


    def build_net(optimizer, dim, epochs, batch, lr=1.0, decay=1.0):
        np.random.seed(1000)
        history = []
        history_00 = []
        Nh = []

        N = Neur.Network([Neur.Dense(vocab_sz, dim), Neur.Dense(dim, vocab_sz)])
        loss_layer = Neur.Ilogit_and_KL(None)
        N_a = Neur.Network([N, loss_layer])

        state = {'k': 0, 'lr': lr, 'decay': decay}
        if optimizer in ['adahessian', 'adam', 'redm', 'spadahessian', 'spadam', 'spredm']:
            state['m'] = np.zeros(N_a.nb_params)
        if optimizer in ['adahessian', 'adam', 'spadahessian', 'spadam']:
            state['v'] = np.zeros(N_a.nb_params)
        if optimizer in ['red', 'redm', 'spredm']:
            state['chat'] = 0.0

        start_time = time.time()
        times = []

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

                if optimizer == 'sgd':
                    state = step_sgd(N_a, state, grads)
                if optimizer == 'red':
                    state = step_red(N_a, state, grads, loss, X_cur)
                if optimizer == 'redm':
                    state = step_redm(N_a, state, grads, loss, X_cur)
                if optimizer == 'adam':
                    state = step_adam(N_a, state, grads)
                if optimizer == 'adahessian':
                    state = step_adahessian(N_a, state, grads, loss_layer, X_cur, Y_cur)
                if optimizer == 'spredm':
                    state = step_spredm(N_a, state, grads, loss, X_cur)
                if optimizer == 'spadam':
                    state = step_spadam(N_a, state, grads)
                if optimizer == 'spadahessian':
                    state = step_spadahessian(N_a, state, grads, loss_layer, X_cur, Y_cur)

                batches += 1

                if i % 50 == 0:
                    batch_loop.set_postfix({"loss": f"{total_loss / batches:.2f}"})

            avg_loss = total_loss / batches
            history.append(avg_loss)

            state['lr'] *= state['decay']
            times.append(time.time() - start_time)
            Nh.append(copy.deepcopy(N))

        return Nh, history, history_00, times

    return (build_net,)


@app.cell
def _(build_net):
    import matplotlib.pyplot as plt
    import itertools

    networks = {}
    loss_history = {}
    grad_history = {}
    runtime = {}

    optimizers = ['sgd', 'red', 'redm', 'adahessian', 'adam']
    for optim in optimizers:
        print(optim)
        if (optim == 'sgd'):
            networks[optim], loss_history[optim], grad_history[optim], runtime[optim] = build_net(optim, 100, 20, 256, 3.0, 1.0)
        else:
            networks[optim], loss_history[optim], grad_history[optim], runtime[optim] = build_net(optim, 100, 20, 256, 0.01)
    return grad_history, loss_history, networks, optimizers, plt, runtime


@app.cell
def _(Neur, X_test, Y_test, get_cbow, get_one_hot, networks, optimizers):
    batch = 256

    for optim2 in optimizers:
        N = networks[optim2][-1]
        loss_layer = Neur.Ilogit_and_KL(None)
        N_a = Neur.Network([N, loss_layer])

        total_loss = 0
        batches = 0

        for k in range(0, len(X_test), batch):
            X_cur = get_cbow(X_test[k:k+batch])
            Y_cur = get_one_hot(Y_test[k:k+batch])

            loss_layer.save_D = Y_cur.T
            loss = N_a.forward(X_cur.T)

            total_loss += loss
            batches += 1

        avg = total_loss / batches
        print(f"{optim2} Test Loss: {avg:.4f}")
    return


@app.cell
def _(
    Neur,
    X_test,
    Y_test,
    get_cbow,
    get_one_hot,
    networks,
    np,
    optimizers,
    plt,
):
    display = {"sgd": "SGD", "red": "RED", "redm": "RED-M", "adahessian": "AdaHessian", "adam": "Adam"}
    def evaluate_and_plot_test_loss():
        test_losses = {optim: [] for optim in optimizers}
        batch_size = 256

        for optim in optimizers:
            for N in networks[optim]:
                loss_layer = Neur.Ilogit_and_KL(None)
                N_a = Neur.Network([N, loss_layer])
                total_loss = 0
                batches = 0
                for k in range(0, len(X_test), batch_size):
                    X_cur = get_cbow(X_test[k : k + batch_size])
                    Y_cur = get_one_hot(Y_test[k : k + batch_size])

                    loss_layer.save_D = Y_cur.T
                    loss = N_a.forward(X_cur.T)

                    total_loss += loss
                    batches += 1

                test_losses[optim].append(total_loss / batches)

        for optim, losses in test_losses.items():
            plt.plot(range(1, 21), losses, label=display[optim], marker="o", markersize=3)

        plt.xlabel("Epochs")
        plt.ylabel("Test Loss")
        plt.title("Test Loss Across 20 Epochs")
        plt.legend()
        plt.xticks(np.arange(0, 21, 2))
        plt.show()
    
    evaluate_and_plot_test_loss()
    return (display,)


@app.cell
def _(grad_history, np, plt):
    x_axis = np.linspace(0, 10, len(grad_history['sgd']))

    plt.plot(x_axis, grad_history['redm'], alpha=0.1, label='RED-M')
    plt.plot(x_axis, grad_history['adam'], alpha=0.1, label='Adam')
    plt.plot(x_axis, grad_history['adahessian'],  alpha=0.1, label='AdaHessian')
    plt.plot(x_axis, grad_history['red'], color='r', alpha=0.1, label='RED')
    plt.plot(x_axis, grad_history['sgd'], color='b', alpha=0.1, label='SGD')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient of W[0][0]')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(loss_history, np, plt, runtime):
    #plt.figure(figsize=(10,6))

    plt.plot(range(1, 21), loss_history['sgd'], marker='o',markersize=3, label=f'SGD ({round(runtime['sgd'][-1])}s)')
    plt.plot(range(1, 21), loss_history['red'], marker='o',markersize=3, label=f'RED ({round(runtime['red'][-1])}s)')
    plt.plot(range(1, 21), loss_history['redm'], marker='o',markersize=3, label=f'RED-M ({round(runtime['redm'][-1])}s)')
    plt.plot(range(1, 21), loss_history['adahessian'], marker='o', markersize=3, label=f'AdaHessian ({round(runtime['adahessian'][-1])}s)')
    plt.plot(range(1, 21), loss_history['adam'], marker='o', markersize=3, label=f'Adam ({round(runtime['adam'][-1])}s)')

    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title("Training Loss Across 20 Epochs")
    plt.xticks(np.arange(0, 21, 2))
    plt.legend()
    plt.show()
    return


@app.cell
def _(dict, networks, np, vocab):
    embed = networks['redm'][-1].list_layers[0].A.T
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


@app.cell
def _(dict, networks, np, optimizers, vocab, vocab_st):
    def run_simlex_benchmark(filepath, norm_embed):
        simlex, model = [], []

        with open(filepath, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().lower().split('\t')

                if len(parts) >= 4 and parts[0] in vocab_st and parts[1] in vocab_st:
                    simlex.append(float(parts[3]))
                    cos_sim = np.dot(norm_embed[dict[parts[0]]], norm_embed[dict[parts[1]]])
                    model.append(cos_sim)

        # spearman rank correlation
        n = len(simlex)
        d_sq = (np.argsort(np.argsort(simlex)) - np.argsort(np.argsort(model))) ** 2
        res = 1 - (6 * np.sum(d_sq)) / (n * (n ** 2 - 1))
        print("Simlex999 benchmark:")
        print(f"Data size: {n}")
        print(f"Spearman's Rho: {res:.4f}\n")

    def run_wordsim_benchmark(filepath, norm_embed):
        wordsim, model = [], []

        with open(filepath, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().lower().split(',')

                if len(parts) >= 3 and parts[0] in vocab_st and parts[1] in vocab_st:
                    wordsim.append(float(parts[2]))
                    cos_sim = np.dot(norm_embed[dict[parts[0]]], norm_embed[dict[parts[1]]])
                    model.append(cos_sim)

        # spearman rank correlation
        n = len(wordsim)
        d_sq = (np.argsort(np.argsort(wordsim)) - np.argsort(np.argsort(model))) ** 2
        res = 1 - (6 * np.sum(d_sq)) / (n * (n ** 2 - 1))

        print("WordSim353 benchmark:")
        print(f"Data size: {n}")
        print(f"Spearman's Rho: {res:.4f}\n")

    def run_google_benchmark(path, norm_embed, k=5):
        tests = []
        valid = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().lower()
                words = line.split()
                if len(words) == 4 and all(w in vocab_st for w in words):
                    tests.append(tuple(words))
                    valid += 1

        top1 = 0
        topk = 0
        for (a, b, c, ans) in tests:
            vec_a = norm_embed[dict[a]]
            vec_b = norm_embed[dict[b]]
            vec_c = norm_embed[dict[c]]

            target = vec_b - vec_a + vec_c
            target = target / np.linalg.norm(target)
            similar = norm_embed @ target
            similar[dict[a]] = -100
            similar[dict[b]] = -100
            similar[dict[c]] = -100
            nearest = np.argsort(similar)[::-1]

            best_words = [vocab[idx] for idx in nearest[:k]]

            if best_words[0] == ans:
                top1 += 1
            if ans in best_words:
                topk += 1

        acc1 = (top1 / valid) * 100
        acck = (topk / valid) * 100

        print("Google benchmark:")
        print(f"Top-1 Accuracy: {top1}/{valid} ({acc1:.2f}%)")
        print(f"Top-{k} Accuracy: {topk}/{valid} ({acck:.2f}%)\n")

    for optim3 in optimizers:
        embed2 = networks[optim3][4].list_layers[0].A.T
        norms2 = np.linalg.norm(embed2, axis=1, keepdims=True)
        norm_embed2 = embed2 / norms2
        print(f"===== {optim3} =====")
        run_simlex_benchmark("SimLex-999.txt", norm_embed2)
        run_wordsim_benchmark("wordsim353crowd.csv", norm_embed2)
        run_google_benchmark("questions-words.txt", norm_embed2, k=3)
        run_google_benchmark("questions-words.txt", norm_embed2, k=5)
    return run_google_benchmark, run_simlex_benchmark, run_wordsim_benchmark


@app.cell
def _():
    # model_save = N
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backup arrays & testing with backups
    """)
    return


@app.cell
def _():
    import json
    import pickle

    return json, pickle


@app.cell
def _(
    dict,
    grad_history,
    json,
    loss_history,
    networks,
    np,
    optimizers,
    pickle,
    runtime,
    vocab,
):
    # backup
    with open("vocab_5.json", "w") as f:
        json.dump({'vocab': vocab, 'dict': dict}, f)

    with open("plot_data_5.pkl", "wb") as f:
        pickle.dump({
            'loss_history': loss_history,
            'grad_history': grad_history,
            'runtime': runtime
        }, f)

    all_embed = {}
    for optim4 in optimizers:
        embed3 = networks[optim4][-1].list_layers[0].A.T
        norms3 = np.linalg.norm(embed3, axis=1, keepdims=True)
        all_embed[optim4] = embed3 / norms3

    np.savez("all_word_vectors_5.npz", **all_embed)
    return


@app.cell
def _(
    np,
    optimizers,
    run_google_benchmark,
    run_simlex_benchmark,
    run_wordsim_benchmark,
):
    loaded = np.load("all_word_vectors_2.npz")
    for optim5 in optimizers:
        norm_embed3 = loaded[optim5]
        print(f"===== {optim5} =====")
        run_simlex_benchmark("SimLex-999.txt", norm_embed3)
        run_wordsim_benchmark("wordsim353crowd.csv", norm_embed3)
        run_google_benchmark("questions-words.txt", norm_embed3, k=3)
        run_google_benchmark("questions-words.txt", norm_embed3, k=5)
    return


@app.cell
def _(display, pickle, plt):
    def loss_over_time():
        with open("plot_data_5.pkl", "rb") as f:
            data = pickle.load(f)
        
        loss_history = data['loss_history']
        runtime = data['runtime']

        for optim, losses in loss_history.items():
            time_data = runtime[optim]
            times = [time_data[i] for i in range(len(losses))]
            plt.plot(times, losses, label=display[optim], marker='o', markersize=3)

        plt.xlabel("Cumulative Time (seconds)")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.show()

    loss_over_time()
    return


@app.cell
def _(
    Neur,
    X_test,
    Y_test,
    display,
    get_cbow,
    get_one_hot,
    networks,
    optimizers,
    pickle,
    plt,
):
    def test_loss_over_time():
        with open("plot_data_5.pkl", "rb") as f:
            data = pickle.load(f)
        runtime = data['runtime']

        test_losses = {optim: [] for optim in optimizers}
        batch_size = 256

        for optim in optimizers:
            for N in networks[optim]:
                loss_layer = Neur.Ilogit_and_KL(None)
                N_a = Neur.Network([N, loss_layer])
                total_loss = 0
                batches = 0
            
                for k in range(0, len(X_test), batch_size):
                    X_cur = get_cbow(X_test[k : k + batch_size])
                    Y_cur = get_one_hot(Y_test[k : k + batch_size])

                    loss_layer.save_D = Y_cur.T
                    loss = N_a.forward(X_cur.T)

                    total_loss += loss
                    batches += 1

                test_losses[optim].append(total_loss / batches)

        for optim, losses in test_losses.items():
            time_data = runtime[optim]
            times = [time_data[i] for i in range(len(losses))]
            plt.plot(times, losses, label=display[optim], marker="o", markersize=3)

        plt.xlabel("Cumulative Time (seconds)")
        plt.ylabel("Test Loss")
        plt.title("Test Loss Over Time")
        plt.legend()
        plt.show()

    test_loss_over_time()
    return
