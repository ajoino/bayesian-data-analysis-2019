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
	//matrix[3, 1] mu;
	real<lower=0> mu_0;
	real<lower=0> mu_1;
	real<upper=0> mu_2;
	matrix[3, 1] phi;
	vector<lower=0>[3] tau;
	real<lower=0> sigma;
	matrix[3, num_users] eta;
}
transformed parameters {
	//matrix[3, num_users] theta;
	row_vector<lower=0>[num_users] lower_bound;
	row_vector[num_users] upper_bound;
	row_vector[num_users] decay;
	lower_bound = mu_0 + phi[1, 1] * is_user_child + eta[1] * tau[1];
	upper_bound = mu_1 + phi[2, 1] * is_user_child + eta[2] * tau[2];
	decay = mu_2 + phi[3, 1] * is_user_child + eta[3] * tau[3];
}
model {
	vector[num_data] log_reaction_summand;
	eta[1] ~ normal(0, 1);
	eta[2] ~ normal(0, 1);
	eta[3] ~ normal(0, 1);
	tau ~ gamma(1, 3);
	sigma ~ gamma(1, 3);
	for (datum in 1:num_data)
		log_reaction_summand[datum] = lower_bound[users[datum]] 
									+ exp(upper_bound[users[datum]] 
										+ decay[users[datum]] * attempt[datum]);
	log_reaction_times ~ normal(log_reaction_summand, sigma);
}
generated quantities {
}
