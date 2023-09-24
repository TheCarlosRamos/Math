import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

nums = [32, 1, 2, 3, 54, 60, 27] 
runs = 1000  
somas = []

for _ in range(runs):
    chosen = random.sample(nums, 5)  
    soma = sum(chosen)  
    somas.append(soma) 

mean = np.mean(somas)  
std = np.std(somas)  

valores = np.random.normal(mean, std, runs)

distribution_names = ['norm', 'expon', 'lognorm', 'gamma', 'beta']

best_fit_name = None
best_fit_params = {}
best_fit_sse = float('inf')

for distribution_name in distribution_names:
    distribution = getattr(stats, distribution_name)
    params = distribution.fit(valores)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    hist, bins = np.histogram(valores, bins=100, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    pdf = distribution.pdf(bin_centers, loc=loc, scale=scale, *arg)
    
    sse = np.sum(np.power(pdf - hist, 2.0))
    
    if best_fit_sse > sse > 0:
        best_fit_name = distribution_name
        best_fit_params = params
        best_fit_sse = sse

plt.hist(valores, bins=100, density=True, alpha=0.7)

best_fit_distribution = getattr(stats, best_fit_name)
pdf = best_fit_distribution.pdf(bin_centers, *best_fit_params[:-2], loc=best_fit_params[-2], scale=best_fit_params[-1])
plt.plot(bin_centers, pdf, 'r-', label=best_fit_name)

mu = best_fit_params[-2]
sigma = best_fit_params[-1]

equation = r'$f(x; \mu={:.2f}, \sigma={:.2f}) = \frac{{1}}{{{:.2f} \sqrt{{2\pi}}}} e^{{-\frac{{(x-{:.2f})^2}}{{2{:.2f}^2}}}}$'.format(mu, sigma, sigma, mu, sigma)
plt.text(0.6, 0.8, equation, fontsize=12, transform=plt.gca().transAxes)

plt.xlabel('Soma dos Valores')
plt.ylabel('Densidade')
plt.title('Histograma com Melhor Ajuste')
plt.legend()

plt.show()
