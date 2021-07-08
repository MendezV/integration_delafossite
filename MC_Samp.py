import numpy as np
import matplotlib.pyplot as plt
def nasty_function(x):
    x_0 = 3.0
    a = 0.01
    return np.exp(-(x**2))/((x-x_0)**2 + a**2)
x_walk = np.empty((0)) #this is an empty list to keep all the steps
x_0 = 8.0*((np.random.random())-0.5) #this is the initialization
x_walk = np.append(x_walk,x_0)
print(x_walk)

n_iterations = 200000 #this is the number of iterations I want to make
for i in range(n_iterations):
    x_prime = np.random.normal(x_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    alpha = nasty_function(x_prime)/nasty_function(x_walk[i])
    if(alpha>=1.0):
        x_walk  = np.append(x_walk,x_prime)
    else:
        beta = np.random.random()
        if(beta<=alpha):
            x_walk = np.append(x_walk,x_prime)
        else:
            x_walk = np.append(x_walk,x_walk[i])
x=np.linspace(-4,4,100)
f = nasty_function(x)
norm = sum(f*(x[1]-x[0]))
#plot(x,f/norm, linewidth=1, color='r')
count, bins = np.histogram(x_walk, bins=1000)
plt.bar(bins[:-1], count, width=bins[1]-bins[0])
#fig = figure(1, figsize=(9.5,6.5))
plt.xlabel('x')
plt.ylabel('p(x)')
plt.show()


def nasty_function2(x,y):
    x_0 = 3.0
    a = 0.01
    return np.exp(-(x**2+y**2))/((x-x_0)**2 + a**2)


x_walk = np.empty((0)) #this is an empty list to keep all the steps
y_walk = np.empty((0)) #this is an empty list to keep all the steps
x_0 = 8.0*((np.random.random())-0.5) #this is the initialization
y_0 = 8.0*((np.random.random())-0.5) #this is the initialization
x_walk = np.append(x_walk,x_0)
y_walk = np.append(y_walk,y_0)
print(x_walk,y_walk)


n_iterations = 400000 #this is the number of iterations I want to make
for i in range(n_iterations):
    x_prime = np.random.normal(x_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    y_prime = np.random.normal(y_walk[i], 0.1) #0.1 is the sigma in the normal distribution
    alpha = nasty_function2(x_prime,y_prime)/nasty_function2(x_walk[i],y_walk[i])
    if(alpha>=1.0):
        x_walk  = np.append(x_walk,x_prime)
        y_walk  = np.append(y_walk,y_prime)
    else:
        beta = np.random.random()
        if(beta<=alpha):
            x_walk  = np.append(x_walk,x_prime)
            y_walk  = np.append(y_walk,y_prime)
        else:
            x_walk = np.append(x_walk,x_walk[i])
            y_walk = np.append(y_walk,y_walk[i])

plt.scatter(x_walk,y_walk, s=1)
plt.show()

# create data
x = np.random.normal(size=50000)
y = x * 3 + np.random.normal(size=50000)

# Big bins
plt.hist2d(x_walk,y_walk, bins=(50, 50), cmap=plt.cm.jet)
plt.show()

# Small bins
plt.hist2d(x_walk,y_walk, bins=(300, 300), cmap=plt.cm.jet)
plt.show()

# If you do not set the same values for X and Y, the bins won't be a square!
plt.hist2d(x_walk,y_walk, bins=(300, 30), cmap=plt.cm.jet)
plt.show()
