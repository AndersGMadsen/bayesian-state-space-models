import casadi

def kalman_casadi_opt_linear(states, y, A, Q, H, R):
    
        # x_{t+1} = A x_t + q_t
        # y_{t+1} = H x_{t+1} + r_{t+1}
    
        # J = sum(r^2) + sum(q^2)

        def diff(x, A):
            return x[: ,1:] - A @ x[:, :-1]
    
        opti = casadi.Opti()
    
        # Define the symbolic variables
        n_steps = states.shape[0] 
        n_states =  states.shape[1] # 4
        n_measurements = y.shape[1]
    
        # make casadi variables
        y = casadi.MX(y.T)
        A = casadi.MX(A)
        Q = casadi.MX(Q)
        H = casadi.MX(H)
        R = casadi.MX(R)
    
        # Define the decision variables
        x = opti.variable(n_states, n_steps)
        q = opti.variable(n_states, n_steps-1)
        r = opti.variable(n_measurements, n_steps)
    
        #V = casadi.sumsqr(q) + casadi.sumsqr(r) # V = sum(r^2) + sum(q^2)

        v1 = casadi.sumsqr(casadi.mtimes(casadi.inv(Q), q))
        v2 = casadi.sumsqr(casadi.mtimes(casadi.inv(R), r))
        
        V = v1 + v2
        
        # V = x.T R^{-1} x + y.T Q^{-1} y
        #V = casadi.sumsqr(q @ casadi.inv(Q) @ q.T) + casadi.sumsqr(r @ casadi.inv(R) @ r.T)

        opti.subject_to(diff(x, A) == q) # x_{t+1} = A x_t + q_t
        opti.subject_to(y - H @ x == r) # y_{t+1} = H x_{t+1} + r_{t+1}
    
        opti.minimize(V)
    
        opti.solver('ipopt')

        sol = opti.solve()
    
        return sol.value(x), sol.value(q), sol.value(r)


def kalman_casadi_opt_nonlinear(states, y, f, h, Q, R):
    
        # x_{t+1} = f(x_t) + q_t
        # y_{t+1} = h(x_{t+1}) + r_{t+1}
    
        # J = sum(r^2) + sum(q^2)

        def diff(x, f):
            return x[:, 1:] - f(x[:, :-1], ca=True)
    
        opti = casadi.Opti()
    
        # Define the symbolic variables
        n_steps = states.shape[0] 
        n_states =  states.shape[1] # 4
        n_measurements = y.shape[1]
    
        # make casadi variables
        y = casadi.MX(y.T)
        Q = casadi.MX(Q)
        R = casadi.MX(R)
    
        # Define the decision variables
        x = opti.variable(n_states, n_steps)
        q = opti.variable(n_states, n_steps-1)
        r = opti.variable(n_measurements, n_steps)

        v1 = casadi.sumsqr(casadi.mtimes(casadi.inv(Q), q))
        v2 = casadi.sumsqr(casadi.mtimes(casadi.inv(R), r))

        V = v1 + v2

        opti.subject_to(diff(x, f) == q) # x_{t+1} = f(x_t) + q_t
        opti.subject_to(y - h(x, ca=True) == r) # y_{t+1} = h(x_{t+1}) + r_{t+1}

        opti.minimize(V)
    
        opti.solver('ipopt')
    
        sol = opti.solve()
    
        return sol.value(x), sol.value(q), sol.value(r)