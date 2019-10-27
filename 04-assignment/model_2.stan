data {
    int<lower=0> N;
    int y[N];
}
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ beta(18.25,6.75);
    y ~ bernoulli(theta);
}
