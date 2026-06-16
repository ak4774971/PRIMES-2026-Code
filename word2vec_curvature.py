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
    return data, vocab_sz


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
    return Neur, X_train, Y_train, np, window


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
def _(
    Neur,
    X_train,
    Y_train,
    get_cbow,
    get_one_hot,
    np,
    step_spadahessian,
    step_spadam,
    step_spredm,
    sz,
    tqdm,
    vocab_sz,
    window,
):
    import time
    import copy

    def step_sgd(N_a, state, g):
        params = N_a.get_params()
        N_a.set_params(params - state['lr'] * g)
        state['step_size'] = state['lr']
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

        ck = max(abs((lp + lm - 2 * lv) / (eps ** 2)) / nrm2, 1e-20) + state['lambda']
        state['chat'] = beta3 * state['chat'] + (1 - beta3) * ck
        ctilde = state['chat'] / (1 - beta3 ** (state['k'] + 1))
        Lk = max(ctilde, ck)
        rk = np.dot(d, g) / (2 * nrm2 * Lk)

        N_a.set_params(p - ell * rk * d)
        state['k'] += 1
        state['ck'] = ck
        state['step_size'] = ell * rk
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
        state['step_size'] = np.mean(state['lr'] / (np.sqrt(vh) + e))
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
        state['step_size'] = np.mean(state['lr'] / (np.sqrt(vh) + e))
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

        ck = max(abs((lp + lm - 2 * lv) / (eps ** 2)) / nrm2, 1e-20) + state['lambda']
        state['chat'] = beta3 * state['chat'] + (1 - beta3) * ck
        ctilde = state['chat'] / (1 - beta3 ** (state['k'] + 1))
        Lk = max(ctilde, ck)
        rk = 1 / (2 * Lk)

        p -= ell * rk * d
        N_a.set_params(p)
        state['k'] += 1
        state['ck'] = ck
        state['step_size'] = ell * rk
        return state

    def build_net(optimizer, dim, epochs, batch, lr=1.0, decay=1.0):
        np.random.seed(1000)
        history = []
        history_00 = []
        Nh = []

        N = Neur.Network([Neur.Dense(vocab_sz, dim), Neur.Dense(dim, vocab_sz)])
        loss_layer = Neur.Ilogit_and_KL(None)
        N_a = Neur.Network([N, loss_layer])

        state = {'k': 0, 'lr': lr, 'decay': decay, 'ck': 0, 'lambda': 1e-7}
        if optimizer in ['adahessian', 'adam', 'redm', 'spadam', 'spredm']:
            state['m'] = np.zeros(N_a.nb_params)
        if optimizer in ['adahessian', 'adam', 'spadam']:
            state['v'] = np.zeros(N_a.nb_params)
        if optimizer in ['red', 'redm']:
            state['chat'] = 0.0
        
        start_time = time.time()
        times = []
        steps = []
        curvature = []
        max_b = 0

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
                grads, in_grad = N_a.backward(None)

                p = N_a.get_params()
                loss_reg = loss + state['lambda'] / 2 * np.sum(p ** 2)
                grads += state['lambda'] * p
            
                total_loss += loss_reg
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
                steps.append(state['step_size'])

                if optimizer in ['red', 'redm']:
                    curvature.append(state['ck'])
                    p1 = N_a.get_params()[:vocab_sz * dim]
                    p2 = N_a.get_params()[vocab_sz * dim : vocab_sz * dim * 2]
                    w1 = p1.reshape(vocab_sz, dim)
                    w2 = p2.reshape(vocab_sz, dim)
                    mx1 = np.max(np.linalg.norm(w1, axis=1))
                    mx2 = np.max(np.linalg.norm(w2, axis=1))
                    max_b = max(max_b, mx1, mx2)

                if i % 50 == 0:
                    batch_loop.set_postfix({"loss": f"{total_loss / batches:.2f}"})

            avg_loss = total_loss / batches
            history.append(avg_loss)

            lower = state['lambda'] - 2 * np.sqrt(2 / window)
            upper = state['lambda'] + 2 * (max_b ** 2) * (1 + 1 / (2 * window)) + 2 * np.sqrt(2 / window)

            state['lr'] *= state['decay']
            times.append(time.time() - start_time)
            Nh.append(copy.deepcopy(N))

        return Nh, history, history_00, times, steps, curvature, lower, upper

    return (build_net,)


@app.cell
def _(build_net):
    import matplotlib.pyplot as plt
    import itertools

    networks = {}
    loss_history = {}
    grad_history = {}
    runtime = {}
    step_size = {}
    lower = {}
    upper = {}
    curvature = {}

    optimizers = ['red', 'redm', 'sgd']
    for optim in optimizers:
        print(optim)
        if (optim == 'sgd'):
            networks[optim], loss_history[optim], grad_history[optim], runtime[optim], step_size[optim], curvature[optim], lower[optim], upper[optim] = build_net(optim, 100, 5, 256, 3.0, 1.0)
        else:
            networks[optim], loss_history[optim], grad_history[optim], runtime[optim], step_size[optim], curvature[optim], lower[optim], upper[optim] = build_net(optim, 100, 5, 256, 0.01)
    return curvature, lower, plt, step_size, upper


@app.cell
def _(np, plt, step_size):
    x_axis = np.linspace(0, 5, len(step_size['sgd']))

    plt.plot(x_axis, step_size['redm'], alpha=0.5, lw=0.5, label='RED-M')
    #plt.plot(x_axis, step_size['adam'], alpha=0.5, lw=0.5, label='Adam')
    plt.plot(x_axis, step_size['red'], alpha=0.5, lw=0.5, label='RED')
    plt.plot(x_axis, step_size['sgd'], alpha=0.5, label='SGD')

    plt.xlabel('Epoch')
    plt.ylabel('Step Size')
    plt.yscale('log')
    plt.title('Effective Step Sizes Over Training')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(curvature, lower, np, plt, upper):
    def plot_scatter(optim, name):
        batches = np.arange(len(curvature[optim]))
        plt.scatter(batches, curvature[optim], s=3, color='purple', alpha=0.1)
        plt.axhline(y=lower[optim], color='blue', label='Lower bound')
        plt.plot([], [], color='red', label=f'Upper bound = {upper[optim]:.2f} (off-scale)')
        plt.xlabel('Batch')
        plt.ylabel('Curvature')
        plt.title(f"Local Curvature with Theoretical Bounds for {name}")
        plt.legend()
        plt.show()

    plot_scatter('red', 'RED')
    plot_scatter('redm', 'RED-M')
    return


if __name__ == "__main__":
    app.run()
