# ============================================================================
# Параметрическая модель эндогенного переключения (switching regression), R.
# Бонус к заданию 5.6. Реализует ту же модель, что и пакет switchSelection
# (Коссова, Потанин, 2018, 2022): уравнение отбора (подключения) с инструментом Z
# и два уравнения выручки для режимов D=0 и D=1 с совместной нормальностью ошибок.
# Здесь модель оценена методом максимального правдоподобия на базовом R (optim),
# чтобы код был самодостаточным и воспроизводимым без внешних зависимостей.
#
# Готовый пакет можно применить так:
#   install.packages("switchSelection"); library(switchSelection)
#   model <- msel(D ~ Z + lage + ssize + online + city,
#                 Y ~ lage + ssize + online + city, regimes = c(0, 1), data = dat)
# ============================================================================

set.seed(20260606)

## --- генерация данных, идентичная hw.py (DGP раздела 2) ---
n      <- 10000
U      <- rnorm(n)
age    <- exp(rnorm(n, 1.6, 0.5))
size   <- ceiling(exp(rnorm(n, 1.2, 0.6)))
online <- as.integer(runif(n) < plogis(-0.2 + 0.8 * U))
city   <- as.integer(runif(n) < 0.5)
Z      <- as.integer(runif(n) < 0.5)
g      <- -0.25 * log(age) + 0.35 * sqrt(size) + 0.6 * online + 0.2 * city - 0.2 * online * city
base   <- -1.2 + 0.7 * U + 0.1 * U^2 + g + rnorm(n)
D      <- as.integer(ifelse(Z == 1, base + 0.9 >= 0, base >= 0))
mu     <- 50 + 40 * sqrt(size) + 15 * log(age) + 30 * online + 25 * city + 20 * U + 3 * U^2
tau    <- 35 + 12 * U - 15 * online - 10 * city + 3 * log(age)
Y      <- mu + D * tau + 25 * rnorm(n)

## --- матрицы регрессоров (корректные функциональные формы) ---
X  <- cbind(1, log(age), sqrt(size), online, city)   # уравнения исходов (без Z: условие исключения)
W  <- cbind(X, Z)                                     # уравнение отбора (с инструментом Z)
kf <- ncol(X); kw <- ncol(W)

## --- логарифмическая функция правдоподобия модели эндогенного переключения ---
negll <- function(th) {
  gpar <- th[1:kw]
  b0   <- th[(kw + 1):(kw + kf)]
  b1   <- th[(kw + kf + 1):(kw + 2 * kf)]
  s0   <- exp(th[kw + 2 * kf + 1]); s1 <- exp(th[kw + 2 * kf + 2])
  r0   <- tanh(th[kw + 2 * kf + 3]); r1 <- tanh(th[kw + 2 * kf + 4])
  Wg <- as.vector(W %*% gpar)
  e1 <- (Y - as.vector(X %*% b1)) / s1
  e0 <- (Y - as.vector(X %*% b0)) / s0
  ll1 <- dnorm(e1, log = TRUE) - log(s1) + pnorm((Wg + r1 * e1) / sqrt(1 - r1^2), log.p = TRUE)
  ll0 <- dnorm(e0, log = TRUE) - log(s0) + pnorm(-(Wg + r0 * e0) / sqrt(1 - r0^2), log.p = TRUE)
  ll  <- ifelse(D == 1, ll1, ll0)
  if (any(!is.finite(ll))) return(1e10)
  -sum(ll)
}

## --- стартовые значения: МНК по режимам ---
b1_0 <- coef(lm(Y ~ X[, -1], subset = D == 1))
b0_0 <- coef(lm(Y ~ X[, -1], subset = D == 0))
th0  <- c(rep(0, kw), b0_0, b1_0, log(25), log(25), 0, 0)

fit  <- optim(th0, negll, method = "Nelder-Mead",
              control = list(maxit = 60000, reltol = 1e-9))

## --- извлечение оценок и среднего эффекта ---
th <- fit$par
b0 <- th[(kw + 1):(kw + kf)]
b1 <- th[(kw + kf + 1):(kw + 2 * kf)]
r0 <- tanh(th[kw + 2 * kf + 3]); r1 <- tanh(th[kw + 2 * kf + 4])
eff <- as.vector(X %*% (b1 - b0))
cat(sprintf("rho0 = %.3f, rho1 = %.3f (эндогенность отбора)\n", r0, r1))
cat(sprintf("Параметрическая ATE (switching MLE) = %.2f  (истинный ATE = 27.96)\n", mean(eff)))
