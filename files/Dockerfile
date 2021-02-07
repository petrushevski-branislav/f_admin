ARG PHP_VERSION=7.4
ARG COMPOSER_VERSION=2.0

FROM composer:${COMPOSER_VERSION}
FROM php:${PHP_VERSION}-cli

RUN apt-get update && \
    apt-get install -y autoconf pkg-config libssl-dev git libzip-dev zlib1g-dev supervisor && \
    pecl install mongodb && docker-php-ext-enable mongodb && \
    pecl install xdebug && docker-php-ext-enable xdebug && \
    docker-php-ext-install -j$(nproc) pdo pdo_mysql sockets zip

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord"]

COPY --from=composer /usr/bin/composer /usr/local/bin/composer

WORKDIR /app
COPY . .
RUN composer install
