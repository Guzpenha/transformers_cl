import math

def linear(x, t, c0):
	return (x* ((1-c0)/t)) + c0

def root_2(x, t, c0):
	return ((x* ((1-(c0**2.0))/t)) + (c0**2.0))**(1./2)

def root_5(x, t, c0):
	return ((x* ((1-(c0**5.0))/t)) + (c0**5.0))**(1./5)

def quadratic(x, t, c0):
	return (x* ((1-c0**1.54)/t))**2 + c0

def cubic(x, t, c0):
	return (x* ((1-c0**1.87)/t))**3 + c0

def step(x, t, c0):
	if x <= t*0.33:
		return 0.33 
	elif x> t*0.33 and x<= t*0.66:
		return 0.66 
	else:
		return 1

PACING_FUNCTIONS = {
	'linear': linear,
	'root_2': root_2,
	'root_5': root_5,
	'quadratic': quadratic,
	'cubic': cubic,
	'step': step
}
