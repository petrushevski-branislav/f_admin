#FROM php:7.4-fpm-alpine
ARG PHP_VERSION=7.4
ARG COMPOSER_VERSION=2.0

FROM composer:${COMPOSER_VERSION}
FROM php:${PHP_VERSION}-cli

#RUN docker-php-ext-install pdo pdo_mysql sockets
RUN apt-get update && \
    apt-get install -y autoconf pkg-config libssl-dev git libzip-dev zlib1g-dev && \
    pecl install mongodb && docker-php-ext-enable mongodb && \
    pecl install xdebug && docker-php-ext-enable xdebug && \
    docker-php-ext-install -j$(nproc) pdo pdo_mysql sockets zip

#RUN curl -sS https://getcomposer.org/installer | php -- \
#     --install-dir=/usr/local/bin --filename=composer

COPY --from=composer /usr/bin/composer /usr/local/bin/composer

WORKDIR /app
COPY . .
RUN composer install
