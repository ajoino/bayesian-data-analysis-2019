data {
    int<lower=0> num_users;
	int<lower=0> num_data;
    real<lower=0> reaction_times[num_data];
	int users[num_data];
}
transformed data {
	real log_reaction_times[num_data];
	log_reaction_times = log(reaction_times);
}
parameters {
    real mu;
	real<lower = 0> tau;
	real theta[num_users];
	real<lower = 0> sigma;
}
model {
	sigma ~ uniform(0, 1000);
	tau ~ uniform(0, 1000);
	mu ~ normal(0, 1000);
	theta ~ normal(mu, tau);
	for (datum in 1:num_data)
		log_reaction_times[datum] ~ normal(theta[users[datum]], sigma);
}
generated quantities {
	real exp_theta[num_users];
	real exp_mu;
	exp_theta = exp(theta);
	exp_mu = exp(mu);
}

