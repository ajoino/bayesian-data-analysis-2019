data {
	int<lower=0> num_data;
    real reaction_times[num_data];
}
parameters {
	real theta;
}
model {
	reaction_times ~ normal(theta, 50);
}

