#!/usr/bin/python3

from casadi import DM, SX, vertcat, nlpsol, inf

x = SX.sym('x',3)
p = SX.sym('p',2)
f = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
g = vertcat(
    x[0] - p[0],
    p[1] - x[0]
)

nlp = {
    'x': x,
    'p': p,
    'f': f,
    'g': g
}

solver = nlpsol('nlpsol', 'ipopt', nlp, dict())


res = solver(
    p=DM([-2, -1]),
    x0=DM.zeros((3,1)),
    lbg=0,
    ubg=inf
)

print(res['x'])
