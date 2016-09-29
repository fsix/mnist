def gradient_with_momentum(curr_weight, curr_sample, prev_gm, alpha, beta):
	''' curr_weight is the current estimation of the parameters,
	prev_gm stands for the gradient with momentum of the last 
	iteration. curr_sample is the sample in the training set 
	that we are using at this iteration. curr_g is the gradient 
	without adding the momentum term of the current iteration. 
	alpha is the learning rate and beta is the momentum fraction. '''
	curr_g = gradient(curr_weight, curr_sample)
	curr_gm = alpha*curr_g + beta*prev_gm
	return curr_gm

def gradient(curr_weight, curr_sample):
	gd = derivative(curr_weight, curr_sample)
	return gd