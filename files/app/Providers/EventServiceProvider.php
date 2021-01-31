<?php

namespace App\Providers;

use App\Jobs\TestJob;
use App\Jobs\UserCreated;
use Illuminate\Foundation\Support\Providers\EventServiceProvider as ServiceProvider;

class EventServiceProvider extends ServiceProvider
{
    public function boot()
    {
        \App::bindMethod(TestJob::class . '@handle', function($job) {
            return $job -> handle();
        });

        \App::bindMethod(UserCreated::class . '@handle', function($job) {
            return $job -> handle();
        });
    }
}
