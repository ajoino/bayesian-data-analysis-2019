data {
    int<lower=0> num_users;
	int<lower=0> num_data;
    real<lower=0> reaction_times[num_data];
	int users[num_data];
	int is_datum_child[num_data];
	int is_user_child[num_users];
}
transformed data {
	real log_reaction_times[num_data];
	log_reaction_times = log(reaction_times);
}
parameters {
    real mu[2];
	real<lower=0> tau;
	real theta[num_users];
	real<lower=0> sigma;
	// real u;
	// real<lower=0> v;
}
model {
	//  ~ uniform(0, 1000);
	//  ~ uniform(0, 1000);
	//mu ~ uniform(0, 1000); //normal(u, v);
	//phi ~ uniform(0, 1000);
	//sigma ~ uniform(0, 1000);
	//tau ~ uniform(0, 1000);
	for (user in 1:num_users)
		theta[user] ~ normal(mu[1] + mu[2] * is_user_child[user], tau);
	for (datum in 1:num_data)
		log_reaction_times[datum] ~ normal(theta[users[datum]], sigma);
}
generated quantities {
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
}
