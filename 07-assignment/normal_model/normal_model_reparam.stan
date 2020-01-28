data {
    int<lower=0> num_users;
	int<lower=0> num_data;
    real<lower=0> reaction_times[num_data];
	real<lower=0> attempt[num_data];
	int users[num_data];
	int is_datum_child[num_data];
	int is_user_child[num_users];
}
transformed data {
	real log_reaction_times[num_data];
	log_reaction_times = log(reaction_times);
}
parameters {
    real adult_mean[2];
	real child_correction[2];
	real<lower=0> tau[2];
	real user_intersect[num_users];
	real user_slope[num_users];
	real<lower=0> sigma;
}
model {
	vector[num_users] intersect_summand;
	vector[num_users] slope_summand;
	vector[num_data] log_reaction_summand;
	for (user in 1:num_users) {
		intersect_summand[user] = adult_mean[1] + child_correction[1] * is_user_child[user];
		slope_summand[user] = adult_mean[2] + child_correction[2] * is_user_child[user];
	}
	user_intersect ~ normal(intersect_summand, tau[1]);
	user_slope ~ normal(slope_summand, tau[2]);
	for (datum in 1:num_data)
		log_reaction_summand[datum] = user_intersect[users[datum]] 
									+ user_slope[users[datum]] * attempt[datum];
		//log_reaction_times[datum] ~ normal(theta[users[datum]], sigma);
	log_reaction_times ~ normal(log_reaction_summand, sigma);
}
generated quantities {
	/*
	real theta_median[num_users];
	real theta_mean[num_users];
	real mu_median[2];
	real adult_group_mean;
	real child_group_mean;
	theta_median = exp(theta);
	for (user in 1:num_users)
		theta_mean[user] = exp(theta[user] + sigma*sigma/2);
	mu_median = exp(mu);
	adult_group_mean = exp(mu[1] + sigma*sigma/2 + tau*tau/2);
	child_group_mean = exp(mu[1] + mu[2] + sigma*sigma/2 + tau*tau/2);
	*/
}
