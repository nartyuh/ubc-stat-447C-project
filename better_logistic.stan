data {
  // train data
  int<lower=0> N;
  vector[N] x1;
  vector[N] x2;
  vector[N] x5;
  vector[N] x7;
  vector[N] x9;
  vector[N] x10;
  int<lower=0,upper=1> y[N];
  
  // test data
  int<lower=0> N_new;
  vector[N_new] x1_new;
  vector[N_new] x2_new;
  vector[N_new] x5_new;
  vector[N_new] x7_new;
  vector[N_new] x9_new;
  vector[N_new] x10_new;
}

parameters {
  real b0;
  real b1;
  real b2;
  real b5;
  real b7;
  real b9;
  real b10;
}

model {
  b0 ~ cauchy(0, 10);
  b1 ~ cauchy(0, 2.5);
  b2 ~ cauchy(0, 2.5);
  b5 ~ cauchy(0, 2.5);
  b7 ~ cauchy(0, 2.5);
  b9 ~ cauchy(0, 2.5);
  b10 ~ cauchy(0, 2.5);
  y ~ bernoulli_logit(b0 + b1*x1 + b2*x2 + b5*x5 + b7*x7 + b9*x9 + b10*x10);
}

generated quantities {
  vector[N_new] y_new;
  for (i in 1:N_new) {
    y_new[i] = bernoulli_logit_rng(b0 + b1*x1[i] + b2*x2[i] + b5*x5[i] + b7*x7[i] + b9*x9[i] + b10*x10[i]);
  }
}

