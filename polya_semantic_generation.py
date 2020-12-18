from Polya import PolyaUrn, load_polya
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

init_balls_per_color = 1
init_colors = 5
nu = 2
rho = 3
eta = 0.8
eta = 1.0
entropy_base = 2
seq_len = 200000

p = PolyaUrn(rho=rho, nu=nu, eta=eta, entropy_base=entropy_base, init_colors=init_colors, 
             init_balls_per_color=init_balls_per_color)
p.get_sequence(seq_len=seq_len, get_seq_entropies=True, print_every=200)
p.save()
p.save_extras()

print("--- %s seconds ---" % (time.time() - start_time))