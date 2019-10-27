data {
    int<lower=0> N;
    int y[N];
}
parameters {
    real<lower=0, upper=1> theta;
}
model {
    theta ~ beta(100,100);
    y ~ bernoulli(theta);
}
