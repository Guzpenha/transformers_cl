import math

def linear(x, t, c0):
	return (x* ((1-c0)/t)) + c0

def root_2(x, t, c0):
	return ((x* ((1-(c0**2.0))/t)) + (c0**2.0))**(1./2)

def root_5(x, t, c0):
	return ((x* ((1-(c0**5.0))/t)) + (c0**5.0))**(1./5)

def root_10(x, t, c0):
	return ((x* ((1-(c0**10.0))/t)) + (c0**10.0))**(1./10)

def root_20(x, t, c0):
	return ((x* ((1-(c0**20.0))/t)) + (c0**20.0))**(1./20)

def root_50(x, t, c0):
	return ((x* ((1-(c0**50.0))/t)) + (c0**50.0))**(1./50)

def geom_progression(x, t, c0):
	return 2.0**((x* ((math.log(1,2.0)-math.log(c0,2.0))/t)) +math.log(c0,2.0))

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

def standard_training(x, t, c0):
	return 1

PACING_FUNCTIONS = {
	'linear': linear,
	'root_2': root_2,
	'root_5': root_5,
	'root_10': root_10,
	'root_20': root_20,
	'root_50': root_50,
	'quadratic': quadratic,
	'geom_progression': geom_progression,
	'cubic': cubic,
	'step': step,
	'standard_training': standard_training
}
