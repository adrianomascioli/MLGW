import mlgw
from mlgw.GW_generator import GW_generator as gen
import jax.numpy as jnp
import jax
from functools import partial
import matplotlib.pyplot as plt




model = gen('/home/adriano/works/PE/MLGW/mlgw/TD_models/model_0/')
times = jnp.linspace(-8,0.005, 10000) #time grid: peak of 22 mode at t=0
theta = jnp.ones(3)

@jax.jit
def waveform(theta_, times_):
    h_p,h_c = model.get_WF(theta_,times_,modes=[(4,4)])
    return h_p,h_c

h_plus, h_cross = waveform(theta,times)
print(h_plus,h_cross)

#input_ = jnp.ones((1,10,3))

#print(model(params, input_)[0][0])
#print(type(model(params, input_)[0][0]))

'''
@partial(jax.jit, static_argnums=0)
def get_mode_func(mode_generator, theta, t_grid):
    h_real, h_im = mode_generator.get_mode(theta, t_grid)
    return h_real, h_im
'''
#h_real, h_im = get_mode_func(mode_gen, theta, times)
#print(get_amplitude(mode_gen, theta))
#print(get_phase(mode_gen, theta))
#print(h_real,h_im)
