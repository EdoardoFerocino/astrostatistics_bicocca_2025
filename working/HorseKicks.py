import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np

data = {
    'Number of deaths': [0, 1, 2, 3, 4],
    'Number of groups': [109, 65, 22, 3, 1]
}

df = pd.DataFrame(data)

total_groups = df['Number of groups'].sum()
df['Probability'] = df['Number of groups'] / total_groups

mu = np.average(df['Number of deaths'],weights=df['Number of groups'])#np.sum(df['Number of deaths'] * df['Probability'])
print(f"Estimated lambda (mean number of deaths): {mu:.3f}")

x = df['Number of deaths']
poisson_probs = poisson.pmf(x, mu=mu)

plt.bar(x, df['Probability'], color='mediumseagreen', alpha=0.6, label='Empirical Probability')
plt.plot(x, poisson_probs, 'o-', color='darkorange', label=f'Poisson Distribution (λ={mu:.2f})')
plt.xlabel('Number of deaths')
plt.ylabel('Probability')
plt.title('Empirical vs Poisson Probability Distribution')
plt.xticks(x)
plt.legend()
plt.show()
