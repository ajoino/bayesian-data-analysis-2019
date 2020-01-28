data {
    int<lower=0> num_users;
	int<lower=0> num_data;
    vector<lower=0>[num_data] reaction_times;
	vector<lower=0>[num_data] attempt;
	int users[num_data];
	int is_datum_child[num_data];
	row_vector[num_users] is_user_child;
}
transformed data {
	vector[num_data] log_reaction_times;
	log_reaction_times = log(reaction_times);
}
parameters {
	matrix[2, 2] mu;
	vector<lower=0>[2] tau;
	real<lower=0> sigma;
	matrix[2, num_users] eta;
}
transformed parameters {
	matrix[2, num_users] theta;
	theta[1] = mu[1, 1] + mu[1, 2] * is_user_child + eta[1] * tau[1];
	theta[2] = mu[2, 1] + mu[2, 2] * is_user_child + eta[2] * tau[2];
}
model {
	vector[num_data] log_reaction_summand;
	eta[1] ~ normal(0, 1);
	eta[2] ~ normal(0, 1);
	for (datum in 1:num_data)
		log_reaction_summand[datum] = theta[1, users[datum]] + theta[2, users[datum]] * attempt[datum];
	log_reaction_times ~ normal(log_reaction_summand, sigma);
}
generated quantities {
}

