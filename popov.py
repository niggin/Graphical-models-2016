def decode(s, H, q, schedule = 'parallel', damping = 1, max_iter = 300, tol_beliefs = 1e-4, display = False):
    m, n = H.shape
    mu_from = np.zeros((n, m, 2)) #from vertices
    mu_to = np.zeros((m, n, 2)) #to vertices
    b = np.zeros((n, 2))
    old_beliefs = np.zeros((n, 2))
    e = np.zeros(n)
    e_progress = list()
    stable_beliefs = list()
    
    #initialization
    mu_from[:, :, 0] = (1 - q) * H.T
    mu_from[:, :, 1] = q * H.T
    #mu_from = np.random.uniform(0, 1, (n, m, 2))
    #print mu_from[:, :, 0]
    result = 2
    
    def update_mu_to(j, i):
        delta = mu_from[:, j, 0] - mu_from[:, j, 1]
        delta_p = np.prod([delta[k] for k in np.nonzero(H[j])[0] if k != i])
        p = 0.5 * np.array([1 + delta_p, 1 - delta_p])
        mu_to[j, i, 0] = damping * p[s[j]] + (1 - damping) * mu_to[j, i, 0]
        mu_to[j, i, 1] = damping * p[1 - s[j]] + (1 - damping) * mu_to[j, i, 1]
        
    def update_mu_from(i, j):
        index = [item for item in np.nonzero(H[:, i])[0] if item != j]
        mu_new = np.zeros(2)
        mu_new[0] = np.prod(mu_to[index, i, 0]) * (1 - q)
        mu_new[1] = np.prod(mu_to[index, i, 1]) * q
        mu_new /= np.sum(mu_new)
        mu_from[i, j, 0] = damping * mu_new[0] + (1 - damping) * mu_from[i, j, 0]
        mu_from[i, j, 1] = damping * mu_new[1] + (1 - damping) * mu_from[i, j, 1]
    
    def update_beliefs(i):
        b[i, 0] = (1 - q) * np.prod(mu_to[np.nonzero(H[:, i])[0], i, 0])
        b[i, 1] = q * np.prod(mu_to[np.nonzero(H[:, i])[0], i, 1])
        
    def update_e(b):
        e = np.argmax(b, axis=1)
        result = 2
        e_progress.append(e)
        if display:
            output_info(iter)
        if np.all(H.dot(e) == s):
            result = 0
        if np.linalg.norm(b - old_beliefs) / np.linalg.norm(b) < tol_beliefs:
            result = 1
        return e, result
        
    def output_info(iter):
        print 'iteration:', iter + 1
        print 'e', e
        
    def stable_beliefs_rate(beliefs, old_beliefs):
        diff = np.absolute(beliefs - old_beliefs)
        number_no_stable = len(np.where(diff < tol_beliefs)[0])
        return number_no_stable / len(old_beliefs) / 2.

    for iter in range(max_iter):

        if schedule == 'parallel':
            for j in range(m):
                for i in np.nonzero(H[j])[0]:
                    update_mu_to(j, i)
            old_beliefs = b.copy()
            for i in range(n):
                update_beliefs(i)
                for j in np.nonzero(H[:, i])[0]:
                    update_mu_from(i, j)

        if schedule == 'sequential':
            for i in range(n):
                for j in np.nonzero(H[:, i])[0]:
                    update_mu_to(j, i)
                old_beliefs = b.copy()
                update_beliefs(i)
                for j in np.nonzero(H[:, i])[0]:
                    update_mu_from(i, j)

        stable_beliefs.append(stable_beliefs_rate(b, old_beliefs))
        e, result = update_e(b)
        if result != 2:
            break
    return e, result#, e_progress, stable_beliefs

def estimate_errors(H, q, num_points = 200):
    success = 0.0
    err_bit = 0.0
    err_block = 0.0
    
    m, n = H.shape
    
    for i in range(num_points):
        e = np.random.binomial(1, q, n)
        s = H.dot(e) % 2
        est_e, status = decode(s, H, q)
        if status < 2:
            err_block += 1 - np.all(est_e == e)
            err_bit += np.sum(est_e != e) * 1.0 / n
            success += 1
    
    print err_bit, err_block, success
    
    diver = (num_points - success) / num_points
    err_bit /= success
    err_block /= success
    
    return err_bit, err_block, diver

def make_generator_matrix(H):
    m, n = H.shape
    cur_H = H.copy()
    ind = list()
    need_ind = list()
    for i in range(n):
        j = 0
        while (cur_H[j, i] == 0 or j in ind):
            j += 1
            if j >= m:
                break
        if j >= m:
            continue
        ind.append(j)
        need_ind.append(i)
        for k in range(m):
            if cur_H[k, i] != 0 and k != j:
                cur_H[k] = (cur_H[k] + cur_H[j]) % 2
    res_ind = np.zeros(m)
    res_ind[ind] = need_ind
    g_ind = [i for i in range(n) if i not in res_ind]
    k = n - m
    G = np.zeros((n, k))
    G[g_ind, :] = np.eye(k)
    G[res_ind.astype(int), :] = cur_H[:, g_ind]
    return G, np.array(g_ind).astype(int)