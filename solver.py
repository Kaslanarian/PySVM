import numpy as np


class Solver:
    '''
    We want to solve the optimization problem using SMO:
    ```markdown
    min x^T·Q·x / 2 + px
    s.t. y^T·x = 0, 0≤x_i≤C
    ```

    Parameters
    ----------
    l : 向量长度
    Q : 待优化函数的二次项
    p : 待优化函数的一次项
    y : 限制条件中的y向量
    Cp: 正样本的限制C
    Cn: 负样本的限制C
    '''
    def __init__(self,
                 l: int,
                 Q,
                 p,
                 y,
                 alpha,
                 Cp: float,
                 Cn: float,
                 max_iter: int = 1000) -> None:
        self.l = l
        self.Q = Q
        self.p = p
        self.y = y
        self.C = (Cp, Cn)
        self.alpha = alpha
        self.max_iter = max_iter
        self.f = lambda x: x @ self.Q @ x / 2 + self.p @ x

    def solve(self, verbose=False):
        self.grad = np.copy(self.p)  # 梯度：Qa+p
        self.has_coverage = False  # 收敛标志
        obj = self.f(self.alpha)

        n_iter = 0
        while n_iter < self.max_iter and not self.has_coverage:
            i, j = self.__select_working_set()
            if self.has_coverage:
                continue

            self.__update(i, j)

            obj_new = self.f(self.alpha)
            if abs(obj - obj_new) < 1e-5:
                self.has_coverage = True

            obj = obj_new
            n_iter += 1

            if verbose and n_iter % 100 == 0:
                print("{} iters".format(n_iter))
        if verbose:
            print("optimize with {} iterations, objective value {}".format(
                n_iter,
                obj,
            ))
        if n_iter == self.max_iter:
            print("Not coverage, increase the max_iter")
        return obj

    def get_alpha(self):
        return self.alpha

    def __select_working_set(self):
        Iup = {
            i
            for i in range(self.l)
            if (self.alpha[i] < self.C[0] and self.y[i] == 1) or (
                self.alpha[i] > 0 and self.y[i] == -1)
        }
        Ilow = {
            j
            for j in range(self.l)
            if (self.alpha[j] < self.C[1] and self.y[j] == -1) or (
                self.alpha[j] > 0 and self.y[j] == 1)
        }
        m = max({-self.y[i] * self.grad[i] for i in Iup})
        M = min({-self.y[j] * self.grad[j] for j in Ilow})

        if m - M < 1e-5:
            self.has_coverage = True  # 选不出来违反对，停止迭代
            return None, None

        for i in Iup:
            if -self.y[i] * self.grad[i] == m:
                break

        for j in Ilow:
            if -self.y[j] * self.grad[j] == M:
                break

        # candidate_j = [t for t in Ilow if -self.y[t] * self.grad[t] < m]
        # target = [
        #     -(m + self.y[t] * self.grad[t])**2 /
        #     (self.Q[i, i] + self.Q[t, t] - 2 * self.Q[i, t])
        #     for t in candidate_j
        # ]
        # j = candidate_j[target.index(min(target))]

        return i, j

    def __update(self, i, j):
        alpha = self.alpha
        old_alpha_i, old_alpha_j = alpha[[i, j]]
        C_i, C_j = self.C[int(self.y[i] == -1)], self.C[int(self.y[i] == 1)]

        if self.y[i] != self.y[j]:
            quad_coef = self.Q[i, i] + self.Q[j, j] + 2 * self.Q[i, j]
            if quad_coef <= 0:
                quad_coef = 1e-12
            delta = (-self.grad[i] - self.grad[j]) / quad_coef

            diff = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta

            if diff > 0:
                if (alpha[j] < 0):
                    alpha[j] = 0
                    alpha[i] = diff

            else:
                if (alpha[i] < 0):
                    alpha[i] = 0
                    alpha[j] = -diff

            if diff > C_i - C_j:
                if (alpha[i] > C_i):
                    alpha[i] = C_i
                    alpha[j] = C_i - diff

            else:
                if alpha[j] > C_j:
                    alpha[j] = C_j
                    alpha[i] = C_j + diff
        else:
            quad_coef = self.Q[i, i] + self.Q[j, j] - 2 * self.Q[i, j]
            if quad_coef <= 0:
                quad_coef = 1e-12
            delta = (self.grad[i] - self.grad[j]) / quad_coef

            sum = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta

            if sum > C_i:
                if alpha[i] > C_i:
                    alpha[i] = C_i
                    alpha[j] = sum - C_i

            else:
                if alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = sum

            if sum > C_j:
                if alpha[j] > C_j:
                    alpha[j] = C_j
                    alpha[i] = sum - C_j

            else:
                if alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = sum

        delta_alpha_i = alpha[i] - old_alpha_i
        delta_alpha_j = alpha[j] - old_alpha_j

        self.grad += self.Q[[i, j]].T @ np.array([
            delta_alpha_i,
            delta_alpha_j,
        ])
